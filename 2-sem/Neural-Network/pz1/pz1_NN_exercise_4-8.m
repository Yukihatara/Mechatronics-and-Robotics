%% Задание 4 и 5: Поиск наилучшей АРСС модели и оценка RMSE
clear; clc; close all;

%% 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
load('u.mat');
load('fi.mat');

dt = 0.05;  % шаг исходных данных
dt_model = 0.2;  % шаг модели

% Прореживаем данные (каждый 4-й отсчет)
u_down = u(1:4:end);
fi_down = fi(1:4:end);

% Разделяем данные на обучающую и тестирующую выборки (70% / 30%)
N = length(fi_down);
N_train = floor(0.7 * N);  % 70% для обучения
N_test = N - N_train;      % 30% для тестирования

% Создаем отдельные объекты iddata
data_train = iddata(fi_down(1:N_train), u_down(1:N_train), dt_model);
data_test = iddata(fi_down(N_train+1:end), u_down(N_train+1:end), dt_model);

data_train.OutputName = 'Угол φ';
data_train.InputName = 'Момент u';
data_test.OutputName = 'Угол φ';
data_test.InputName = 'Момент u';

fprintf('=== ДАННЫЕ ===\n');
fprintf('Всего отсчетов после прореживания: %d\n', N);
fprintf('Обучающая выборка: %d отсчетов (%.0f%%)\n', N_train, 100*N_train/N);
fprintf('Тестирующая выборка: %d отсчетов (%.0f%%)\n', N_test, 100*N_test/N);
fprintf('Шаг модели: %.2f с\n', dt_model);

%% 2. ПЕРЕБОР ВСЕХ ВОЗМОЖНЫХ ПОРЯДКОВ НА ОБУЧАЮЩЕЙ ВЫБОРКЕ
% Диапазоны для перебора
na_range = 1:10;  % порядок авторегрессии (m)
nb_range = 1:10;  % порядок по входу (n)
nk_range = 1:10;  % задержка (d)

% Массивы для хранения результатов
N_na = length(na_range);
N_nb = length(nb_range);
N_nk = length(nk_range);
total_models = N_na * N_nb * N_nk;

% Инициализация матриц для хранения критериев
loss_function = zeros(N_na, N_nb, N_nk);  % функция потерь на обучении
AIC_values = zeros(N_na, N_nb, N_nk);     % AIC критерий
FPE_values = zeros(N_na, N_nb, N_nk);     % FPE критерий
rmse_train_matrix = zeros(N_na, N_nb, N_nk);  % RMSE на обучении
rmse_test_matrix = zeros(N_na, N_nb, N_nk);   % RMSE на тесте

fprintf('\n=== ПЕРЕБОР МОДЕЛЕЙ ===\n');
fprintf('Всего моделей для проверки: %d\n', total_models);
fprintf('Перебор: na=1..10, nb=1..10, nk=1..10\n');
fprintf('Идет расчет...\n');

% Счетчик для прогресса
model_count = 0;
tic;

% Циклы по всем параметрам
for na = na_range
    for nb = nb_range
        for nk = nk_range
            model_count = model_count + 1;
            
            % Показываем прогресс каждые 100 моделей
            if mod(model_count, 100) == 0
                fprintf('  Прогресс: %d/%d (%.1f%%), время: %.1f с\n', ...
                    model_count, total_models, 100*model_count/total_models, toc);
            end
            
            % Пробуем оценить модель
            try
                % Структура порядка
                order = [na, nb, nk];
                
                % Оценка модели ARX на обучающей выборке
                model = arx(data_train, order);
                
                % Сохраняем критерии (рассчитанные на обучающей выборке)
                loss_function(na, nb, nk) = model.EstimationInfo.LossFcn;
                AIC_values(na, nb, nk) = aic(model);
                FPE_values(na, nb, nk) = model.EstimationInfo.FPE;
                
                % === ЗАДАНИЕ 5: Расчет RMSE ===
                
                % RMSE на обучающей выборке
                yp_train = predict(model, data_train, 1);
                e_train = data_train.y - yp_train.y;
                rmse_train_matrix(na, nb, nk) = sqrt(mean(e_train.^2));
                
                % RMSE на тестирующей выборке
                yp_test = predict(model, data_test, 1);
                e_test = data_test.y - yp_test.y;
                rmse_test_matrix(na, nb, nk) = sqrt(mean(e_test.^2));
                
            catch
                % Если модель не может быть оценена
                loss_function(na, nb, nk) = inf;
                AIC_values(na, nb, nk) = inf;
                FPE_values(na, nb, nk) = inf;
                rmse_train_matrix(na, nb, nk) = inf;
                rmse_test_matrix(na, nb, nk) = inf;
            end
        end
    end
end

fprintf('Перебор завершен за %.1f с\n', toc);

%% 3. ПОИСК ОПТИМАЛЬНЫХ ПАРАМЕТРОВ ПО РАЗНЫМ КРИТЕРИЯМ

% По функции потерь
[min_loss, idx_loss] = min(loss_function(:));
[na_opt_loss, nb_opt_loss, nk_opt_loss] = ind2sub(size(loss_function), idx_loss);

% По AIC
[min_aic, idx_aic] = min(AIC_values(:));
[na_opt_aic, nb_opt_aic, nk_opt_aic] = ind2sub(size(AIC_values), idx_aic);

% По FPE
[min_fpe, idx_fpe] = min(FPE_values(:));
[na_opt_fpe, nb_opt_fpe, nk_opt_fpe] = ind2sub(size(FPE_values), idx_fpe);

% По RMSE на тесте (можно добавить)
[min_rmse_test, idx_rmse] = min(rmse_test_matrix(:));
[na_opt_rmse, nb_opt_rmse, nk_opt_rmse] = ind2sub(size(rmse_test_matrix), idx_rmse);

fprintf('\n=== ОПТИМАЛЬНЫЕ МОДЕЛИ ===\n');
fprintf('По критерию потерь (Loss Fcn):\n');
fprintf('  na=%d, nb=%d, nk=%d, Loss=%.6f\n', ...
    na_opt_loss, nb_opt_loss, nk_opt_loss, min_loss);
fprintf('  RMSE на обучении: %.6f\n', rmse_train_matrix(na_opt_loss, nb_opt_loss, nk_opt_loss));
fprintf('  RMSE на тесте: %.6f\n', rmse_test_matrix(na_opt_loss, nb_opt_loss, nk_opt_loss));

fprintf('\nПо критерию AIC:\n');
fprintf('  na=%d, nb=%d, nk=%d, AIC=%.2f\n', ...
    na_opt_aic, nb_opt_aic, nk_opt_aic, min_aic);
fprintf('  RMSE на обучении: %.6f\n', rmse_train_matrix(na_opt_aic, nb_opt_aic, nk_opt_aic));
fprintf('  RMSE на тесте: %.6f\n', rmse_test_matrix(na_opt_aic, nb_opt_aic, nk_opt_aic));

fprintf('\nПо критерию FPE:\n');
fprintf('  na=%d, nb=%d, nk=%d, FPE=%.6f\n', ...
    na_opt_fpe, nb_opt_fpe, nk_opt_fpe, min_fpe);
fprintf('  RMSE на обучении: %.6f\n', rmse_train_matrix(na_opt_fpe, nb_opt_fpe, nk_opt_fpe));
fprintf('  RMSE на тесте: %.6f\n', rmse_test_matrix(na_opt_fpe, nb_opt_fpe, nk_opt_fpe));

fprintf('\nПо минимальному RMSE на тесте:\n');
fprintf('  na=%d, nb=%d, nk=%d, RMSE_тест=%.6f\n', ...
    na_opt_rmse, nb_opt_rmse, nk_opt_rmse, min_rmse_test);
fprintf('  RMSE на обучении: %.6f\n', rmse_train_matrix(na_opt_rmse, nb_opt_rmse, nk_opt_rmse));

%% 4. АНАЛИЗ ПЕРЕОБУЧЕНИЯ ДЛЯ ОПТИМАЛЬНОЙ МОДЕЛИ ПО AIC
na_opt = na_opt_aic;
nb_opt = nb_opt_aic;
nk_opt = nk_opt_aic;

fprintf('\n=== АНАЛИЗ ПЕРЕОБУЧЕНИЯ ДЛЯ МОДЕЛИ ARX(%d,%d,%d) ===\n', ...
    na_opt, nb_opt, nk_opt);

rmse_train_opt = rmse_train_matrix(na_opt, nb_opt, nk_opt);
rmse_test_opt = rmse_test_matrix(na_opt, nb_opt, nk_opt);
ratio = rmse_test_opt / rmse_train_opt;

fprintf('RMSE на обучении: %.6f\n', rmse_train_opt);
fprintf('RMSE на тесте: %.6f\n', rmse_test_opt);
fprintf('Отношение тест/обучение: %.3f\n', ratio);

if ratio < 1.1
    fprintf('[good] Модель хорошая, переобучения нет\n');
elseif ratio < 1.5
    fprintf('[Warning] Небольшое переобучение, но допустимо\n');
else
    fprintf('[BAD] Сильное переобучение! Модель слишком сложная\n');
end

%% 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ПЕРЕБОРА

% Первая фигура - влияние параметров на Loss и AIC
figure;

% --- ВЛИЯНИЕ ПАРАМЕТРА na (порядок авторегрессии) ---
subplot(1,3,1);
na_mean_loss = squeeze(mean(mean(loss_function, 3), 2));
na_mean_AIC = squeeze(mean(mean(AIC_values, 3), 2));

plot(na_range, na_mean_loss, 'b-o', 'LineWidth', 2); hold on;
plot(na_range, na_mean_AIC/10, 'r-s', 'LineWidth', 2);
xlabel('Порядок авторегрессии na');
ylabel('Значение критерия');
title('Влияние параметра na');
legend('Loss', 'AIC/10', 'Location', 'best');
grid on;

% --- ВЛИЯНИЕ ПАРАМЕТРА nb (порядок по входу) ---
subplot(1,3,2);
nb_mean_loss = squeeze(mean(mean(loss_function, 3), 1));
nb_mean_AIC = squeeze(mean(mean(AIC_values, 3), 1));

plot(nb_range, nb_mean_loss, 'b-o', 'LineWidth', 2); hold on;
plot(nb_range, nb_mean_AIC/10, 'r-s', 'LineWidth', 2);
xlabel('Порядок по входу nb');
ylabel('Значение критерия');
title('Влияние параметра nb');
legend('Loss', 'AIC/10', 'Location', 'best');
grid on;

% --- ВЛИЯНИЕ ПАРАМЕТРА nk (задержка) ---
subplot(1,3,3);
nk_mean_loss = squeeze(mean(mean(loss_function, 2), 1));
nk_mean_AIC = squeeze(mean(mean(AIC_values, 2), 1));

plot(nk_range, nk_mean_loss, 'b-o', 'LineWidth', 2); hold on;
plot(nk_range, nk_mean_AIC/10, 'r-s', 'LineWidth', 2);
xlabel('Задержка nk');
ylabel('Значение критерия');
title('Влияние параметра nk');
legend('Loss', 'AIC/10', 'Location', 'best');
grid on;

% Вторая фигура - влияние параметров на FPE
figure;

% --- ВЛИЯНИЕ ПАРАМЕТРА na на FPE ---
subplot(1,3,1);
na_mean_FPE = squeeze(mean(mean(FPE_values, 3), 2));
plot(na_range, na_mean_FPE, 'g-o', 'LineWidth', 2);
xlabel('Порядок авторегрессии na');
ylabel('Значение FPE');
title('Влияние параметра na на FPE');
grid on;

% --- ВЛИЯНИЕ ПАРАМЕТРА nb на FPE ---
subplot(1,3,2);
nb_mean_FPE = squeeze(mean(mean(FPE_values, 3), 1));
plot(nb_range, nb_mean_FPE, 'g-o', 'LineWidth', 2);
xlabel('Порядок по входу nb');
ylabel('Значение FPE');
title('Влияние параметра nb на FPE');
grid on;

% --- ВЛИЯНИЕ ПАРАМЕТРА nk на FPE ---
subplot(1,3,3);
nk_mean_FPE = squeeze(mean(mean(FPE_values, 2), 1));
plot(nk_range, nk_mean_FPE, 'g-o', 'LineWidth', 2);
xlabel('Задержка nk');
ylabel('Значение FPE');
title('Влияние параметра nk на FPE');
grid on;

% Третья фигура - сравнение при na=nb
figure;

% Берем модели с na=nb от 1 до 10 при nk=1
simple_orders = 1:10;
loss_simple = zeros(1,10);
aic_simple = zeros(1,10);
fpe_simple = zeros(1,10);

for i = 1:10
    loss_simple(i) = loss_function(i, i, 1);
    aic_simple(i) = AIC_values(i, i, 1);
    fpe_simple(i) = FPE_values(i, i, 1);
end

plot(simple_orders, loss_simple, 'b-o', 'LineWidth', 2); hold on;
plot(simple_orders, aic_simple/10, 'r-s', 'LineWidth', 2);
plot(simple_orders, fpe_simple*1000, 'g-^', 'LineWidth', 2);
xlabel('Порядок модели (na=nb)');
ylabel('Значение критерия');
title('Сравнение критериев при na=nb, nk=1');
legend('Loss', 'AIC/10', 'FPE×1000', 'Location', 'best');
grid on;

%% 6. ДЕТАЛЬНЫЙ АНАЛИЗ ОПТИМАЛЬНОЙ МОДЕЛИ (по AIC)

fprintf('\n=== ДЕТАЛЬНЫЙ АНАЛИЗ МОДЕЛИ ARX(%d,%d,%d) ===\n', ...
    na_opt, nb_opt, nk_opt);

% Оцениваем оптимальную модель на ВСЕЙ обучающей выборке
model_opt = arx(data_train, [na_opt, nb_opt, nk_opt]);

% Выводим параметры модели
fprintf('\nПараметры модели:\n');
disp(model_opt);

% Проверка устойчивости
fprintf('\nПроверка устойчивости:\n');
poles = roots(model_opt.A);
fprintf('Полюса модели:\n');
for i = 1:length(poles)
    fprintf('  p%d = %.4f', i, poles(i));
    if abs(poles(i)) < 1
        fprintf(' (устойчив)\n');
    else
        fprintf(' (НЕУСТОЙЧИВ!)\n');
    end
end

%% 7. ГРАФИКИ ДЛЯ ОПТИМАЛЬНОЙ МОДЕЛИ

figure('Position', [100, 100, 1400, 900]);

% Предсказание на обучающей выборке
yp_train = predict(model_opt, data_train, 1);
e_train = data_train.y - yp_train.y;
t_train = (0:N_train-1)' * dt_model;

% Предсказание на тестирующей выборке
yp_test = predict(model_opt, data_test, 1);
e_test = data_test.y - yp_test.y;
t_test = (N_train:N-1)' * dt_model;

% График 1: Обучающая выборка
subplot(2,2,1);
plot(t_train, data_train.y, 'b-', 'LineWidth', 1.5); hold on;
plot(t_train, yp_train.y, 'r--', 'LineWidth', 1.5);
xlabel('Время (с)');
ylabel('φ (рад)');
title(sprintf('Обучающая выборка (RMSE=%.6f)', sqrt(mean(e_train.^2))));
legend('Реальный', 'Модель');
grid on;

% График 2: Тестирующая выборка
subplot(2,2,2);
plot(t_test, data_test.y, 'b-', 'LineWidth', 1.5); hold on;
plot(t_test, yp_test.y, 'r--', 'LineWidth', 1.5);
xlabel('Время (с)');
ylabel('φ (рад)');
title(sprintf('Тестирующая выборка (RMSE=%.6f)', sqrt(mean(e_test.^2))));
legend('Реальный', 'Модель');
grid on;

% График 3: Ошибка на обучении
subplot(2,2,3);
plot(t_train, e_train, 'k-', 'LineWidth', 1);
xlabel('Время (с)');
ylabel('Ошибка');
title(sprintf('Ошибка на обучении (RMSE=%.6f)', sqrt(mean(e_train.^2))));
grid on;

% График 4: Ошибка на тесте
subplot(2,2,4);
plot(t_test, e_test, 'k-', 'LineWidth', 1);
xlabel('Время (с)');
ylabel('Ошибка');
title(sprintf('Ошибка на тесте (RMSE=%.6f)', sqrt(mean(e_test.^2))));
grid on;
%% 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
% Сохраняем результаты перебора
save('model_selection_results.mat', 'na_range', 'nb_range', 'nk_range', ...
     'loss_function', 'AIC_values', 'FPE_values', ...
     'rmse_train_matrix', 'rmse_test_matrix', ...
     'na_opt_loss', 'nb_opt_loss', 'nk_opt_loss', ...
     'na_opt_aic', 'nb_opt_aic', 'nk_opt_aic', ...
     'na_opt_fpe', 'nb_opt_fpe', 'nk_opt_fpe', ...
     'na_opt_rmse', 'nb_opt_rmse', 'nk_opt_rmse');

%% 9. ИТОГОВЫЕ ВЫВОДЫ

fprintf('\n=== ИТОГОВЫЕ ВЫВОДЫ ===\n');
fprintf('1. Формально оптимальная модель по AIC: ARX(%d,%d,%d)\n', na_opt, nb_opt, nk_opt);
fprintf('2. RMSE на обучении: %.6f\n', rmse_train_opt);
fprintf('3. RMSE на тесте: %.6f\n', rmse_test_opt);
fprintf('4. Отношение тест/обучение: %.3f - ', ratio);
if ratio < 1.1
    fprintf('переобучения нет\n');
elseif ratio < 1.5
    fprintf('небольшое переобучение\n');
else
    fprintf('сильное переобучение\n');
end

%% 8. ГРАФИКИ, ПОДТВЕРЖДАЮЩИЕ АДЕКВАТНОСТЬ МОДЕЛИ (ЗАДАНИЕ 8)

% 1. График остатков во времени
figure;
plot(t_test, e_test, 'b-', 'LineWidth', 1);
xlabel('Время (с)');
ylabel('Остатки δ(t)');
title('Остатки модели во времени');
grid on;
yline(0, 'k--');

figure;
% 1. Автокорреляция ошибки (тест)
subplot(2,3,1);
[e_corr, lags] = xcorr(e_test, 50, 'coeff');
lags_time = lags * dt_model;
stem(lags_time, e_corr, 'b', 'LineWidth', 1);
xlabel('Лаг (с)');
ylabel('Корреляция');
title('Автокорреляция ошибки (тест)');
grid on;
xline(0, 'k--');
yline(1.96/sqrt(length(e_test)), 'r--', '95%');
yline(-1.96/sqrt(length(e_test)), 'r--');

% 2. Взаимная корреляция ошибки с входом
subplot(2,3,2);
[e_u_corr, lags] = xcorr(e_test, data_test.u, 50, 'coeff');
stem(lags_time, e_u_corr, 'b', 'LineWidth', 1);
xlabel('Лаг (с)');
ylabel('Корреляция');
title('Корреляция ошибки с входом u(t)');
grid on;
xline(0, 'k--');
yline(1.96/sqrt(length(e_test)), 'r--', '95%');
yline(-1.96/sqrt(length(e_test)), 'r--');

% 3. Гистограмма ошибки
subplot(2,3,3);
histogram(e_test, 30, 'Normalization', 'pdf', 'FaceColor', 'b', 'EdgeColor', 'k');
hold on;
x = linspace(min(e_test), max(e_test), 100);
mu = mean(e_test);
sigma = std(e_test);
y_norm = normpdf(x, mu, sigma);
plot(x, y_norm, 'r-', 'LineWidth', 2);
xlabel('Ошибка');
ylabel('Плотность');
title('Распределение ошибки (тест)');
legend('Ошибка', 'Нормальное');
grid on;

% 4. QQ-plot (проверка нормальности)
subplot(2,3,4);
qqplot(e_test);
title('QQ-plot ошибки');
grid on;

% 5. Прогноз на 3 шага вперед
subplot(2,3,5);
yp_3 = predict(model_opt, data_test, 3);
rmse_3 = sqrt(mean((data_test.y - yp_3.y).^2));
plot(t_test, data_test.y, 'b-', 'LineWidth', 1.5); hold on;
plot(t_test, yp_3.y, 'r--', 'LineWidth', 1.5);
xlabel('Время (с)');
ylabel('φ (рад)');
title(sprintf('Прогноз на 3 шага (RMSE=%.4f)', rmse_3));
legend('Реальный', 'Прогноз');
grid on;

% 6. Прогноз на 5 шагов вперед
subplot(2,3,6);
yp_5 = predict(model_opt, data_test, 5);
rmse_5 = sqrt(mean((data_test.y - yp_5.y).^2));
plot(t_test, data_test.y, 'b-', 'LineWidth', 1.5); hold on;
plot(t_test, yp_5.y, 'r--', 'LineWidth', 1.5);
xlabel('Время (с)');
ylabel('φ (рад)');
title(sprintf('Прогноз на 5 шагов (RMSE=%.4f)', rmse_5));
legend('Реальный', 'Прогноз');
grid on;

% 9. СТАТИСТИЧЕСКИЕ ТЕСТЫ ДЛЯ ПОДТВЕРЖДЕНИЯ АДЕКВАТНОСТИ

fprintf('\n=== ЗАДАНИЕ 8: АДЕКВАТНОСТЬ МОДЕЛИ ===\n');

% Тест на "белый шум" для ошибки
[e_corr_test, lags_test] = xcorr(e_test, 20, 'coeff');
conf_level = 1.96/sqrt(length(e_test));
outside_bounds = sum(abs(e_corr_test(abs(lags_test)>0)) > conf_level);

if outside_bounds == 0
    fprintf('✓ Ошибка - белый шум (корреляция в пределах дов. интервала)\n');
else
    fprintf('⚠ Ошибка имеет %d выбросов за пределами дов. интервала\n', outside_bounds);
end

% Тест на независимость от входа
[e_u_corr_test, ~] = xcorr(e_test, data_test.u, 20, 'coeff');
outside_bounds_u = sum(abs(e_u_corr_test) > conf_level);

if outside_bounds_u == 0
    fprintf('✓ Ошибка не коррелирует с входом (независима)\n');
else
    fprintf('⚠ Ошибка коррелирует с входом в %d точках\n', outside_bounds_u);
end

% Тест на нормальность (критерий Лиллиефорса)
[h, p] = lillietest(e_test);
if h == 0
    fprintf('✓ Ошибка распределена нормально (p=%.3f)\n', p);
else
    fprintf('⚠ Ошибка не является нормальной (p=%.3f)\n', p);
end

% Тест Дарбина-Уотсона на автокорреляцию
dw = sum(diff(e_test).^2) / sum(e_test.^2);
fprintf('Статистика Дарбина-Уотсона: %.3f (ближе к 2 - лучше)\n', dw);

% 10. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
% Сохраняем оптимальную модель
save('optimal_model.mat', 'model_opt');

% Сохраняем результаты перебора
save('model_selection_results.mat', 'na_range', 'nb_range', 'nk_range', ...
     'loss_function', 'AIC_values', 'FPE_values', ...
     'rmse_train_matrix', 'rmse_test_matrix', ...
     'na_opt_loss', 'nb_opt_loss', 'nk_opt_loss', ...
     'na_opt_aic', 'nb_opt_aic', 'nk_opt_aic', ...
     'na_opt_fpe', 'nb_opt_fpe', 'nk_opt_fpe', ...
     'na_opt_rmse', 'nb_opt_rmse', 'nk_opt_rmse');

% 11. ИТОГОВЫЕ ВЫВОДЫ

fprintf('\n=== ИТОГОВЫЕ ВЫВОДЫ ===\n');
fprintf('1. Оптимальная модель по AIC: ARX(%d,%d,%d)\n', na_opt, nb_opt, nk_opt);
fprintf('2. RMSE на обучении: %.6f\n', rmse_train_opt);
fprintf('3. RMSE на тесте: %.6f\n', rmse_test_opt);
fprintf('4. Отношение тест/обучение: %.3f - ', ratio);
if ratio < 1.1
    fprintf('переобучения нет\n');
elseif ratio < 1.5
    fprintf('небольшое переобучение\n');
else
    fprintf('сильное переобучение\n');
end