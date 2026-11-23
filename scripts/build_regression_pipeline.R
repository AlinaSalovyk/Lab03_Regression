# ==============================================================================
# Лабораторна робота №3. Основи регресійного моделювання
# Автор: Садовик Аліна (група АтаВДвR-11)
# Пайплайн навчання, крос-валідації, діагностики та оцінки регресійних моделей
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(scales)
  library(cowplot)
  library(broom)
  library(mgcv)
})

cat(">>> Крок 1. Завантаження даних та фіксація генератора випадкових чисел...\n")
set.seed(42)

df_all <- read.csv("data/processed/clean_data.csv", stringsAsFactors = FALSE)
cat(sprintf("Завантажено очищений датасет: %d спостережень, %d колонок.\n", nrow(df_all), ncol(df_all)))

# Відбір предикторів, доступних на момент прогнозу (без Data Leakage)
# Виключаємо: at_risk, performance_tier, pass_rate, needs_help_score, needs_help_band
features_list <- c(
  "admission_grade", "gpa_score_avg", "first_assessment_score",
  "attendance_rate", "lms_engagement_score", "lms_active_days_ratio", "missed_submissions_count",
  "stress_level_scale", "burnout_score", "sleep_hours_per_day", "psych_distress_index",
  "socioeconomic_deprivation_index", "tuition_up_to_date", "scholarship_holder", "first_generation"
)

# Перевірка на пропущені значення у предикторах та цілі
df_clean <- df_all %>%
  select(student_id, source_dataset, final_score_pct, all_of(features_list), 
         stress_band_factor, tuition_factor, scholarship_factor) %>%
  na.omit()

cat(sprintf("Вибірка після перевірки повних даних: %d спостережень.\n", nrow(df_clean)))

# ==============================================================================
# Крок 2. Розбиття вибірки на Train (80%) та Test (20%)
# ==============================================================================
cat(">>> Крок 2. Розбиття на Train (80%) / Test (20%)...\n")
n_total <- nrow(df_clean)
train_indices <- sample(seq_len(n_total), size = floor(0.80 * n_total))

train_df <- df_clean[train_indices, ]
test_df  <- df_clean[-train_indices, ]

cat(sprintf("Train set: %d спостережень (%.1f%%)\n", nrow(train_df), nrow(train_df)/n_total*100))
cat(sprintf("Test set:  %d спостережень (%.1f%%)\n", nrow(test_df), nrow(test_df)/n_total*100))

# Перевірка збігу розподілу цілі між Train та Test
split_check <- data.frame(
  Split = c("Train (80%)", "Test (20%)", "Full Dataset"),
  N = c(nrow(train_df), nrow(test_df), n_total),
  Mean_Score = c(mean(train_df$final_score_pct), mean(test_df$final_score_pct), mean(df_clean$final_score_pct)),
  SD_Score = c(sd(train_df$final_score_pct), sd(test_df$final_score_pct), sd(df_clean$final_score_pct)),
  Median_Score = c(median(train_df$final_score_pct), median(test_df$final_score_pct), median(df_clean$final_score_pct)),
  IQR_Score = c(IQR(train_df$final_score_pct), IQR(test_df$final_score_pct), IQR(df_clean$final_score_pct))
)
write.csv(split_check, "outputs/tables/train_test_split_validation.csv", row.names = FALSE)
cat("Збережено outputs/tables/train_test_split_validation.csv\n")

# ==============================================================================
# Крок 3. 5-кратна перехресна валідація (5-Fold Cross-Validation) на Train set
# ==============================================================================
cat(">>> Крок 3. Виконання 5-кратної крос-валідації на навчальній вибірці...\n")
k_folds <- 5
folds <- sample(rep(1:k_folds, length.out = nrow(train_df)))

calc_metrics <- function(actual, predicted) {
  rmse_val <- sqrt(mean((actual - predicted)^2))
  mae_val  <- mean(abs(actual - predicted))
  ss_res   <- sum((actual - predicted)^2)
  ss_tot   <- sum((actual - mean(actual))^2)
  r2_val   <- 1 - (ss_res / ss_tot)
  c(RMSE = rmse_val, MAE = mae_val, R2 = r2_val)
}

cv_results <- list()

for (fold in 1:k_folds) {
  cat(sprintf("  -> Обробка фолду %d/%d...\n", fold, k_folds))
  train_cv <- train_df[folds != fold, ]
  val_cv   <- train_df[folds == fold, ]
  
  # 1. Baseline Model (Константний орієнтир: середнє)
  mean_train_fold <- mean(train_cv$final_score_pct)
  pred_m0 <- rep(mean_train_fold, nrow(val_cv))
  
  # 2. Model 1 (Базова множинна лінійна регресія OLS)
  m1_formula <- as.formula(paste("final_score_pct ~", paste(features_list, collapse = " + ")))
  fit_m1_cv <- lm(m1_formula, data = train_cv)
  pred_m1 <- predict(fit_m1_cv, newdata = val_cv)
  
  # 3. Model 2 (Модифікована OLS з взаємодіями та поліномом відвідуваності)
  m2_formula <- as.formula(paste(
    "final_score_pct ~",
    paste(setdiff(features_list, c("attendance_rate", "gpa_score_avg", "psych_distress_index", "tuition_up_to_date", "scholarship_holder")), collapse = " + "),
    "+ gpa_score_avg * psych_distress_index",
    "+ poly(attendance_rate, 2, raw = TRUE)",
    "+ tuition_up_to_date * scholarship_holder"
  ))
  fit_m2_cv <- lm(m2_formula, data = train_cv)
  pred_m2 <- predict(fit_m2_cv, newdata = val_cv)
  
  # 4. Model 3 (Нелінійна узагальнена адитивна модель GAM via bam)
  gam_formula <- as.formula(
    "final_score_pct ~ s(gpa_score_avg, k=5) + s(attendance_rate, k=5) + s(first_assessment_score, k=5) + 
     s(admission_grade, k=5) + s(lms_engagement_score, k=5) + s(psych_distress_index, k=5) + 
     s(sleep_hours_per_day, k=5) + missed_submissions_count + lms_active_days_ratio + 
     stress_level_scale + burnout_score + socioeconomic_deprivation_index + 
     tuition_up_to_date + scholarship_holder + first_generation"
  )
  fit_m3_cv <- bam(gam_formula, data = train_cv, discrete = TRUE)
  pred_m3 <- predict(fit_m3_cv, newdata = val_cv)
  
  # Збір метрик
  res_m0 <- calc_metrics(val_cv$final_score_pct, pred_m0)
  res_m1 <- calc_metrics(val_cv$final_score_pct, pred_m1)
  res_m2 <- calc_metrics(val_cv$final_score_pct, pred_m2)
  res_m3 <- calc_metrics(val_cv$final_score_pct, pred_m3)
  
  cv_results[[fold]] <- rbind(
    data.frame(Fold = fold, Model = "0. Baseline (Mean)", t(res_m0)),
    data.frame(Fold = fold, Model = "1. Multiple Linear (OLS)", t(res_m1)),
    data.frame(Fold = fold, Model = "2. OLS + Interactions & Poly", t(res_m2)),
    data.frame(Fold = fold, Model = "3. Non-linear GAM (Splines)", t(res_m3))
  )
}

cv_df <- do.call(rbind, cv_results)

cv_summary <- cv_df %>%
  group_by(Model) %>%
  summarise(
    RMSE_mean = mean(RMSE),
    RMSE_sd   = sd(RMSE),
    MAE_mean  = mean(MAE),
    MAE_sd    = sd(MAE),
    R2_mean   = mean(R2),
    R2_sd     = sd(R2),
    .groups   = "drop"
  ) %>%
  arrange(RMSE_mean)

write.csv(cv_summary, "outputs/tables/model_cv_comparison.csv", row.names = FALSE)
cat("Збережено outputs/tables/model_cv_comparison.csv\n")

# ==============================================================================
# Крок 4. Навчання фінальних моделей на повній навчальній вибірці (Full Train)
# ==============================================================================
cat(">>> Крок 4. Навчання фінальних моделей на 100% Train set...\n")

# 1. Baseline
baseline_mean <- mean(train_df$final_score_pct)

# 2. Model 1 (Базова OLS)
fit_m1 <- lm(m1_formula, data = train_df)
saveRDS(fit_m1, "outputs/models/model1_ols.rds")

# 3. Model 2 (Взаємодії + Поліноми)
fit_m2 <- lm(m2_formula, data = train_df)
saveRDS(fit_m2, "outputs/models/model2_interaction.rds")

# 4. Model 3 (GAM)
fit_m3 <- bam(gam_formula, data = train_df, discrete = TRUE)
saveRDS(fit_m3, "outputs/models/model3_gam.rds")

# Оцінка коефіцієнтів Model 1 (OLS) з 95% довірчими інтервалами
ols_tidy <- tidy(fit_m1, conf.int = TRUE, conf.level = 0.95) %>%
  mutate(
    term_clean = case_when(
      term == "(Intercept)" ~ "Константа (Intercept)",
      term == "admission_grade" ~ "Вступний бал (admission_grade)",
      term == "gpa_score_avg" ~ "Середній бал GPA (gpa_score_avg)",
      term == "first_assessment_score" ~ "Перший контрольний тест",
      term == "attendance_rate" ~ "Відвідуваність (%)",
      term == "lms_engagement_score" ~ "Індекс взаємодії з LMS",
      term == "lms_active_days_ratio" ~ "Частка активних днів у семестрі",
      term == "missed_submissions_count" ~ "Кількість пропущених робіт",
      term == "stress_level_scale" ~ "Шкала стресу (0-10)",
      term == "burnout_score" ~ "Індекс вигорання (burnout)",
      term == "sleep_hours_per_day" ~ "Тривалість сну (год/добу)",
      term == "psych_distress_index" ~ "Зведений індекс дистресу",
      term == "socioeconomic_deprivation_index" ~ "Індекс депривації (1-10)",
      term == "tuition_up_to_date" ~ "Вчасна сплата за навчання (1/0)",
      term == "scholarship_holder" ~ "Наявність стипендії (1/0)",
      term == "first_generation" ~ "Студент 1-го покоління (1/0)",
      TRUE ~ term
    ),
    significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01  ~ "**",
      p.value < 0.05  ~ "*",
      TRUE ~ "н/з"
    )
  )

write.csv(ols_tidy, "outputs/tables/ols_coefficients_summary.csv", row.names = FALSE)
cat("Збережено outputs/tables/ols_coefficients_summary.csv\n")

# ==============================================================================
# Крок 5. Фінальна незалежна оцінка на тестовій вибірці (Test Set)
# ==============================================================================
cat(">>> Крок 5. Фінальне оцінювання на відкладеному тесті...\n")

test_preds <- test_df %>%
  mutate(
    pred_m0 = baseline_mean,
    pred_m1 = predict(fit_m1, newdata = test_df),
    pred_m2 = predict(fit_m2, newdata = test_df),
    pred_m3 = as.numeric(predict(fit_m3, newdata = test_df))
  )

test_eval <- data.frame(
  Model = c(
    "0. Baseline (Mean Only)",
    "1. Multiple Linear Regression (OLS)",
    "2. OLS + Non-linear & Interactions",
    "3. Generalized Additive Model (GAM)"
  ),
  RMSE = c(
    sqrt(mean((test_preds$final_score_pct - test_preds$pred_m0)^2)),
    sqrt(mean((test_preds$final_score_pct - test_preds$pred_m1)^2)),
    sqrt(mean((test_preds$final_score_pct - test_preds$pred_m2)^2)),
    sqrt(mean((test_preds$final_score_pct - test_preds$pred_m3)^2))
  ),
  MAE = c(
    mean(abs(test_preds$final_score_pct - test_preds$pred_m0)),
    mean(abs(test_preds$final_score_pct - test_preds$pred_m1)),
    mean(abs(test_preds$final_score_pct - test_preds$pred_m2)),
    mean(abs(test_preds$final_score_pct - test_preds$pred_m3))
  ),
  R2 = c(
    1 - sum((test_preds$final_score_pct - test_preds$pred_m0)^2) / sum((test_preds$final_score_pct - mean(test_preds$final_score_pct))^2),
    1 - sum((test_preds$final_score_pct - test_preds$pred_m1)^2) / sum((test_preds$final_score_pct - mean(test_preds$final_score_pct))^2),
    1 - sum((test_preds$final_score_pct - test_preds$pred_m2)^2) / sum((test_preds$final_score_pct - mean(test_preds$final_score_pct))^2),
    1 - sum((test_preds$final_score_pct - test_preds$pred_m3)^2) / sum((test_preds$final_score_pct - mean(test_preds$final_score_pct))^2)
  ),
  Max_Abs_Error = c(
    max(abs(test_preds$final_score_pct - test_preds$pred_m0)),
    max(abs(test_preds$final_score_pct - test_preds$pred_m1)),
    max(abs(test_preds$final_score_pct - test_preds$pred_m2)),
    max(abs(test_preds$final_score_pct - test_preds$pred_m3))
  )
)

write.csv(test_eval, "outputs/tables/test_evaluation_summary.csv", row.names = FALSE)
cat("Збережено outputs/tables/test_evaluation_summary.csv\n")
print(test_eval)

# ==============================================================================
# Крок 6. Побудова аналітичних візуалізацій регресії
# ==============================================================================
cat(">>> Крок 6. Генерація графіків діагностики та якості...\n")

# Підготовка вибірки для швидкої та якісної діагностики (10,000 точок для чіткості графіків)
set.seed(42)
diag_sample <- train_df %>%
  sample_n(15000) %>%
  mutate(
    fitted_m1 = predict(fit_m1, newdata = .),
    resid_m1  = final_score_pct - fitted_m1,
    std_resid_m1 = resid_m1 / sd(resid_m1)
  )

# --- Графік 9: Комплексна діагностика залишків базової моделі (4 панелі) ---
p9a <- ggplot(diag_sample, aes(x = fitted_m1, y = resid_m1)) +
  geom_point(alpha = 0.15, color = "#264653", size = 1) +
  geom_hline(yintercept = 0, color = "#e76f51", linetype = "dashed", linewidth = 0.8) +
  geom_smooth(method = "loess", color = "#2a9d8f", se = FALSE, linewidth = 1) +
  labs(
    title = "A. Залишки vs Прогнозовані значення",
    x = "Прогнозований підсумковий бал (%)",
    y = "Залишки (Residuals, %)"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 11))

p9b <- ggplot(diag_sample, aes(x = resid_m1)) +
  geom_histogram(aes(y = after_stat(density)), bins = 45, fill = "#a8dadc", color = "#457b9d", alpha = 0.8) +
  stat_function(fun = dnorm, args = list(mean = mean(diag_sample$resid_m1), sd = sd(diag_sample$resid_m1)),
                color = "#e63946", linewidth = 1) +
  labs(
    title = "B. Розподіл залишків vs Нормальний",
    x = "Величина залишку (%)",
    y = "Густина імовірності"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 11))

p9c <- ggplot(diag_sample, aes(x = fitted_m1, y = sqrt(abs(std_resid_m1)))) +
  geom_point(alpha = 0.15, color = "#2a9d8f", size = 1) +
  geom_smooth(method = "loess", color = "#e76f51", se = FALSE, linewidth = 1) +
  labs(
    title = "C. Scale-Location (Гомоскедастичність)",
    x = "Прогнозований бал (%)",
    y = "sqrt(|Стандартизовані залишки|)"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 11))

# Q-Q plot
p9d <- ggplot(diag_sample, aes(sample = std_resid_m1)) +
  stat_qq(alpha = 0.2, color = "#1d3557", size = 1) +
  stat_qq_line(color = "#e63946", linetype = "dashed", linewidth = 1) +
  labs(
    title = "D. Нормальний графік Q-Q",
    x = "Теоретичні квантилі",
    y = "Стандартизовані залишки"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 11))

fig09 <- plot_grid(p9a, p9b, p9c, p9d, ncol = 2)
title_fig09 <- ggdraw() + draw_label("Діагностика залишків базової множинної лінійної регресії (OLS)", fontface = "bold", size = 14)
final_fig09 <- plot_grid(title_fig09, fig09, ncol = 1, rel_heights = c(0.06, 1))

ggsave("outputs/figures/fig09_residuals_diagnostics.png", final_fig09, width = 11, height = 8.5, dpi = 300, bg = "white")
cat("Збережено outputs/figures/fig09_residuals_diagnostics.png\n")

# --- Графік 10: Фактичні vs Прогнозовані значення на тестовій вибірці (Test Set) ---
test_sample <- test_preds %>% sample_n(20000)

p10 <- ggplot(test_sample, aes(x = pred_m2, y = final_score_pct)) +
  geom_point(alpha = 0.15, color = "#1d3557", size = 1) +
  geom_abline(intercept = 0, slope = 1, color = "#e63946", linetype = "solid", linewidth = 1.1) +
  geom_smooth(method = "lm", color = "#2a9d8f", linetype = "dashed", se = FALSE, linewidth = 1) +
  annotate("label", x = 10, y = 90, label = sprintf("Test RMSE: %.2f%%\nTest MAE: %.2f%%\nTest R²: %.3f", 
                                                    test_eval$RMSE[3], test_eval$MAE[3], test_eval$R2[3]),
           hjust = 0, size = 4.2, fontface = "bold", fill = "#f1faee") +
  labs(
    title = "Фактичні та прогнозовані значення на відкладеній тестовій вибірці (Model 2)",
    subtitle = "Червона лінія — ідеальний прогноз (y = x); зелена пунктирна — фактичний емпіричний тренд",
    x = "Прогнозований підсумковий бал (%)",
    y = "Фактичний підсумковий бал (%)"
  ) +
  coord_cartesian(xlim = c(0, 100), ylim = c(0, 100)) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray30")
  )

ggsave("outputs/figures/fig10_actual_vs_predicted.png", p10, width = 8.5, height = 6.5, dpi = 300, bg = "white")
cat("Збережено outputs/figures/fig10_actual_vs_predicted.png\n")

# --- Графік 11: Forest plot коефіцієнтів OLS з 95% довірчими інтервалами ---
coef_plot_df <- ols_tidy %>%
  filter(term != "(Intercept)") %>%
  mutate(
    effect_type = ifelse(estimate > 0, "Позитивний ефект", "Негативний ефект"),
    term_clean = reorder(term_clean, estimate)
  )

p11 <- ggplot(coef_plot_df, aes(x = estimate, y = term_clean, color = effect_type)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.25, linewidth = 0.9) +
  geom_point(size = 2.8) +
  scale_color_manual(values = c("Позитивний ефект" = "#2a9d8f", "Негативний ефект" = "#e76f51")) +
  labs(
    title = "Оцінки коефіцієнтів базової регресії (OLS) з 95% довірчими інтервалами",
    subtitle = "Показує маргінальний внесок предиктора в підсумковий бал (%) за фіксації інших факторів",
    x = "Оцінка коефіцієнта (Beta, зміна балу на одиницю предиктора)",
    y = NULL,
    color = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9.5, color = "gray30"),
    legend.position = "bottom"
  )

ggsave("outputs/figures/fig11_coefficients_intervals.png", p11, width = 10, height = 6.5, dpi = 300, bg = "white")
cat("Збережено outputs/figures/fig11_coefficients_intervals.png\n")

# --- Графік 12: Порівняння моделей за 5-Fold CV (RMSE та MAE) ---
cv_long <- cv_summary %>%
  select(Model, RMSE_mean, RMSE_sd, MAE_mean, MAE_sd) %>%
  tidyr::pivot_longer(
    cols = c(RMSE_mean, MAE_mean),
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  mutate(
    sd = ifelse(Metric == "RMSE_mean", RMSE_sd, MAE_sd),
    Metric_Label = ifelse(Metric == "RMSE_mean", "RMSE (Середньоквадратична похибка)", "MAE (Середня абсолютна похибка)")
  )

p12 <- ggplot(cv_long, aes(x = Model, y = Value, fill = Model)) +
  geom_col(width = 0.55, show.legend = FALSE, alpha = 0.85) +
  geom_errorbar(aes(ymin = Value - sd, ymax = Value + sd), width = 0.2, linewidth = 0.8, color = "black") +
  geom_text(aes(label = sprintf("%.2f%%", Value)), vjust = -1.2, size = 3.6, fontface = "bold") +
  facet_wrap(~Metric_Label, scales = "free_y") +
  scale_fill_manual(values = c(
    "0. Baseline (Mean)" = "#b0bec5",
    "1. Multiple Linear (OLS)" = "#457b9d",
    "2. OLS + Interactions & Poly" = "#2a9d8f",
    "3. Non-linear GAM (Splines)" = "#1d3557"
  )) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Порівняння точності моделей на 5-кратній крос-валідації (5-Fold CV)",
    subtitle = "Похибки наведені у відсоткових пунктах (менше значення = краща точність). Вуса показують ±1 SD за фолдами.",
    x = NULL,
    y = "Похибка прогнозу (%)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
    plot.subtitle = element_text(size = 9.5, hjust = 0.5, color = "gray30"),
    axis.text.x = element_text(angle = 20, hjust = 1, face = "bold"),
    strip.text = element_text(face = "bold", size = 11)
  )

ggsave("outputs/figures/fig12_cv_metrics_comparison.png", p12, width = 10, height = 5.5, dpi = 300, bg = "white")
cat("Збережено outputs/figures/fig12_cv_metrics_comparison.png\n")

# --- Графік 13: Аналіз похибок прогнозу у розрізі субгруп (Error Analysis) ---
test_preds_err <- test_preds %>%
  mutate(
    error = final_score_pct - pred_m2,
    abs_error = abs(error),
    attendance_decile = cut(attendance_rate, breaks = c(0, 50, 70, 85, 100), 
                            labels = c("<50% (Критична)", "50-70% (Низька)", "70-85% (Середня)", ">85% (Висока)")),
    stress_band = factor(case_when(
      stress_level_scale <= 3 ~ "Низький стрес (0-3)",
      stress_level_scale <= 6 ~ "Помірний стрес (4-6)",
      stress_level_scale <= 8 ~ "Високий стрес (7-8)",
      TRUE ~ "Екстремальний (9-10)"
    ), levels = c("Низький стрес (0-3)", "Помірний стрес (4-6)", "Високий стрес (7-8)", "Екстремальний (9-10)"))
  )

p13a <- ggplot(test_preds_err, aes(x = attendance_decile, y = error, fill = attendance_decile)) +
  geom_boxplot(outlier.alpha = 0.05, outlier.size = 0.5, show.legend = FALSE) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "A. Похибка за категоріями відвідуваності", x = "Категорія відвідуваності", y = "Залишок: y - y_hat (%)") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold"))

p13b <- ggplot(test_preds_err, aes(x = stress_band, y = error, fill = stress_band)) +
  geom_boxplot(outlier.alpha = 0.05, outlier.size = 0.5, show.legend = FALSE) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_fill_brewer(palette = "Reds") +
  labs(title = "B. Похибка за рівнем стресу", x = "Рівень стресу", y = "Залишок: y - y_hat (%)") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold"), axis.text.x = element_text(angle = 15, hjust = 1))

p13c <- ggplot(test_preds_err, aes(x = source_dataset, y = error, fill = source_dataset)) +
  geom_boxplot(outlier.alpha = 0.05, outlier.size = 0.5, show.legend = FALSE) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  scale_fill_brewer(palette = "Dark2") +
  labs(title = "C. Похибка між когортами (Domain Shift)", x = "Первинна когорта даних", y = "Залишок: y - y_hat (%)") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold"), axis.text.x = element_text(angle = 15, hjust = 1))

fig13 <- plot_grid(p13a, p13b, p13c, ncol = 3)
title_fig13 <- ggdraw() + draw_label("Аналіз похибок прогнозування моделі у підгрупах студентів (Test Set)", fontface = "bold", size = 13)
final_fig13 <- plot_grid(title_fig13, fig13, ncol = 1, rel_heights = c(0.08, 1))

ggsave("outputs/figures/fig13_error_analysis_subgroups.png", final_fig13, width = 12, height = 4.8, dpi = 300, bg = "white")
cat("Збережено outputs/figures/fig13_error_analysis_subgroups.png\n")

cat(">>> Регресійний пайплайн успішно виконано!\n")
