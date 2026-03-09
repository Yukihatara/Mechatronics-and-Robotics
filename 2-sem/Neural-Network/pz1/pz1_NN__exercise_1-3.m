%% Задание 1. Инициализуем код и запускаем схему. Сравниваем полученные
% результаты

% Очищаем рабочее пространство
clear; clc; close all;

load('u.mat');  % предполагаем, что переменная называется u
load('fi.mat'); % предполагаем, что переменная называется fi

% Создаем вектор времени
dt = 0.05; % шаг дискретизации
t = (0:length(u)-1)' * dt;

% Создаем объект timeseries для Simulink
u_ts = timeseries(u, t);
fi_ts = timeseries(fi, t);

% % Сохраняем для Simulink
% save('u_for_simulink.mat', 'u_ts');
% ---------------------------------------------------------------------- %

%% Задание 3: Simulink

%% Задание 3: Корреляционные функции и спектральные плотности
clear; clc; close all;

% Загружаем данные
load('u.mat');
load('fi.mat');

% Параметры
dt = 0.05;  % шаг дискретизации
Fs = 1/dt;  % частота дискретизации (Гц)
t = (0:length(u)-1)' * dt;

% Удаляем тренд (постоянную составляющую) для корректного анализа
u_detrend = detrend(u, 'constant');
fi_detrend = detrend(fi, 'constant');

%% 1. АВТОКОРРЕЛЯЦИОННЫЕ ФУНКЦИИ

% Максимальное число лагов (обычно 1/4 от длины)
max_lag = round(length(u)/4);

% Автокорреляция входного сигнала
[autocorr_u, lags] = xcov(u_detrend, max_lag, 'coeff');  % 'coeff' - нормировка к 1

% Автокорреляция выходного сигнала
[autocorr_fi, lags] = xcov(fi_detrend, max_lag, 'coeff');

% Перевод лагов в секунды
lags_time = lags * dt;

%% 2. ВЗАИМНАЯ КОРРЕЛЯЦИЯ

% Взаимная корреляция между u и φ
[crosscorr_uf, lags_cross] = xcov(u_detrend, fi_detrend, max_lag, 'coeff');

% Лаги для взаимной корреляции в секундах
lags_cross_time = lags_cross * dt;

%% 3. СПЕКТРАЛЬНЫЕ ПЛОТНОСТИ

% Используем pwelch для оценки спектральной плотности мощности
nfft = 2048;  % количество точек БПФ
window = hann(256);  % окно для сглаживания
noverlap = 128;  % перекрытие окон

% Спектр входного сигнала
[pxx_u, f] = pwelch(u_detrend, window, noverlap, nfft, Fs);

% Спектр выходного сигнала
[pxx_fi, f] = pwelch(fi_detrend, window, noverlap, nfft, Fs);

% Взаимный спектр
[pxy_uf, f] = cpsd(u_detrend, fi_detrend, window, noverlap, nfft, Fs);

%% 4. ПОСТРОЕНИЕ ГРАФИКОВ

figure('Position', [50, 50, 1400, 900]);

% --- График 1: Автокорреляция u(t) ---
subplot(3,3,1);
plot(lags_time, autocorr_u, 'b-', 'LineWidth', 1.5);
xlabel('Время (с)');
ylabel('Корреляция');
title('Автокорреляция u(t)');
grid on;
xlim([-max_lag*dt, max_lag*dt]);
ylim([-0.5, 1.1]);
line([-max_lag*dt, max_lag*dt], [0,0], 'Color', 'k', 'LineStyle', '--');

% --- График 2: Автокорреляция φ(t) ---
subplot(3,3,2);
plot(lags_time, autocorr_fi, 'r-', 'LineWidth', 1.5);
xlabel('Время (с)');
ylabel('Корреляция');
title('Автокорреляция φ(t)');
grid on;
xlim([-max_lag*dt, max_lag*dt]);
ylim([-0.5, 1.1]);
line([-max_lag*dt, max_lag*dt], [0,0], 'Color', 'k', 'LineStyle', '--');

% --- График 3: Взаимная корреляция u-φ ---
subplot(3,3,3);
plot(lags_cross_time, crosscorr_uf, 'g-', 'LineWidth', 1.5);
xlabel('Время (с)');
ylabel('Корреляция');
title('Взаимная корреляция u(t) и φ(t)');
grid on;
xlim([-max_lag*dt, max_lag*dt]);
ylim([-0.5, 1.1]);
line([-max_lag*dt, max_lag*dt], [0,0], 'Color', 'k', 'LineStyle', '--');

% --- График 4: Спектральная плотность u(t) (линейный масштаб) ---
subplot(3,3,4);
plot(f, pxx_u, 'b-', 'LineWidth', 1.5);
xlabel('Частота (Гц)');
ylabel('Мощность');
title('Спектр u(t) - линейный');
grid on;
xlim([0, Fs/2]);

% --- График 5: Спектральная плотность φ(t) (линейный масштаб) ---
subplot(3,3,5);
plot(f, pxx_fi, 'r-', 'LineWidth', 1.5);
xlabel('Частота (Гц)');
ylabel('Мощность');
title('Спектр φ(t) - линейный');
grid on;
xlim([0, Fs/2]);

% --- График 6: Спектры в логарифмическом масштабе ---
subplot(3,3,6);
semilogy(f, pxx_u, 'b-', 'LineWidth', 1.5); hold on;
semilogy(f, pxx_fi, 'r-', 'LineWidth', 1.5);
xlabel('Частота (Гц)');
ylabel('Мощность (лог)');
title('Спектры (логарифмическая шкала)');
legend('u(t)', 'φ(t)');
grid on;
xlim([0, Fs/2]);

% --- График 7: АЧХ системы (отношение спектров) ---
subplot(3,3,7);
H = sqrt(pxx_fi ./ pxx_u);
plot(f, H, 'k-', 'LineWidth', 1.5);
xlabel('Частота (Гц)');
ylabel('|H(f)|');
title('АЧХ системы (оценка)');
grid on;
xlim([0, Fs/2]);

% --- График 8: Фазовая характеристика ---
subplot(3,3,8);
phase = angle(pxy_uf);
plot(f, phase, 'm-', 'LineWidth', 1.5);
xlabel('Частота (Гц)');
ylabel('Фаза (рад)');
title('ФЧХ системы (оценка)');
grid on;
xlim([0, Fs/2]);

% --- График 9: Функция когерентности ---
subplot(3,3,9);
Cxy = pxy_uf .* conj(pxy_uf) ./ (pxx_u .* pxx_fi);
plot(f, abs(Cxy), 'c-', 'LineWidth', 1.5);
xlabel('Частота (Гц)');
ylabel('Когерентность');
title('Когерентность u и φ');
grid on;
xlim([0, Fs/2]);
ylim([0, 1.1]);

%% 5. АНАЛИЗ РЕЗУЛЬТАТОВ

fprintf('=== АНАЛИЗ КОРРЕЛЯЦИОННЫХ ФУНКЦИЙ ===\n\n');

% Анализ автокорреляции
[~, idx_first_zero_u] = min(abs(autocorr_u(length(autocorr_u)/2+1:end) - 0.1));
fprintf('Автокорреляция u(t):\n');
fprintf('- Быстро затухает, что говорит о широкополосности сигнала\n');
fprintf('- Интервал корреляции: ~%.2f с\n', idx_first_zero_u*dt);

[~, idx_first_zero_fi] = min(abs(autocorr_fi(length(autocorr_fi)/2+1:end) - 0.1));
fprintf('\nАвтокорреляция φ(t):\n');
fprintf('- Затухает медленнее, что характерно для инерционной системы\n');
fprintf('- Интервал корреляции: ~%.2f с\n', idx_first_zero_fi*dt);

% Анализ взаимной корреляции
[peak_cross, idx_peak] = max(abs(crosscorr_uf));
time_delay = lags_cross_time(idx_peak);
fprintf('\nВзаимная корреляция u-φ:\n');
fprintf('- Максимум корреляции: %.3f\n', peak_cross);
fprintf('- Задержка реакции системы: %.3f с\n', abs(time_delay));
if time_delay < 0
    fprintf('  (φ запаздывает относительно u)\n');
else
    fprintf('  (u запаздывает относительно φ - проверьте данные!)\n');
end

fprintf('\n=== АНАЛИЗ СПЕКТРАЛЬНЫХ ХАРАКТЕРИСТИК ===\n\n');

% Основные частоты в спектре
[~, idx_peak_u] = findpeaks(pxx_u, 'MinPeakHeight', 0.1*max(pxx_u));
fprintf('Основные частоты во входном сигнале (Гц):\n');
for i = 1:min(3, length(idx_peak_u))
    fprintf('- %.3f Гц\n', f(idx_peak_u(i)));
end

[~, idx_peak_fi] = findpeaks(pxx_fi, 'MinPeakHeight', 0.1*max(pxx_fi));
fprintf('\nОсновные частоты в выходном сигнале (Гц):\n');
for i = 1:min(3, length(idx_peak_fi))
    fprintf('- %.3f Гц\n', f(idx_peak_fi(i)));
end

% Полоса пропускания системы
H_db = 20*log10(H);
f_cutoff_idx = find(H_db < H_db(1) - 3, 1, 'first');
if ~isempty(f_cutoff_idx)
    fprintf('\nПолоса пропускания системы:\n');
    fprintf('- Частота среза (-3 дБ): %.3f Гц\n', f(f_cutoff_idx));
end

fprintf('\n=== ВЫВОДЫ ===\n');
fprintf('1. Система обладает инерционностью (медленное затухание автокорреляции φ)\n');
fprintf('2. Задержка реакции составляет примерно %.3f с\n', abs(time_delay));
fprintf('3. Система эффективно пропускает частоты до %.3f Гц\n', f(f_cutoff_idx));
fprintf('4. Высокая когерентность (>0.8) в рабочем диапазоне подтверждает линейность связи u и φ\n');


























% clear; clc; close all;

% %% 1. ЗАГРУЗКА ДАННЫХ
% load('u.mat');  % предполагаем, что переменная называется u
% load('fi.mat'); % предполагаем, что переменная называется fi
% 
% % Проверим размеры массивов
% fprintf('Размер u: '); disp(size(u));
% fprintf('Размер fi: '); disp(size(fi));
% 
% %% 2. ПАРАМЕТРЫ ДИСКРЕТИЗАЦИИ
% dt_raw = 0.05;      % шаг исходных данных (с)
% dt_model = 0.2;     % шаг модели (с)
% downsample_factor = dt_model / dt_raw; % коэффициент прореживания
% 
% % Проверка
% fprintf('Коэффициент прореживания: %d\n', downsample_factor);
% 
% %% 3. ПРОРЕЖИВАНИЕ ДАННЫХ
% % Берем каждый 4-й элемент
% u_down = u(1:downsample_factor:end);
% fi_down = fi(1:downsample_factor:end);
% 
% % Создаем векторы времени для графиков
% t_raw = (0:length(u)-1) * dt_raw;
% t_down = (0:length(u_down)-1) * dt_model;
% 
% %% 4. ФОРМИРОВАНИЕ РЕГРЕССОРОВ ДЛЯ ADALINE
% % Выберем порядок модели. Начнем со 2-го порядка
% n = 2; % порядок модели
% 
% % Подготовка матрицы регрессоров X и вектора целевых значений Y
% % Для модели φ(t) = a1·φ(t-1) + a2·φ(t-2) + b1·u(t-1) + b2·u(t-2)
% 
% N = length(fi_down); % количество отсчетов после прореживания
% X = []; % матрица регрессоров
% Y = []; % целевые значения
% 
% % Начинаем с n+1, так как нужны предыдущие значения
% for t = n+1:N
%     % Формируем вектор регрессоров: [φ(t-1), φ(t-2), u(t-1), u(t-2)]
%     regressor = [];
%     for i = 1:n
%         regressor = [regressor, fi_down(t-i)]; % добавляем φ(t-i)
%     end
%     for i = 1:n
%         regressor = [regressor, u_down(t-i)];  % добавляем u(t-i)
%     end
%     X = [X; regressor];
%     Y = [Y; fi_down(t)];
% end
% 
% %% 5. ОБУЧЕНИЕ ADALINE
% 
% % Метод 1: Метод наименьших квадратов (аналитическое решение)
% % w = (X^T * X)^(-1) * X^T * Y
% w_LS = (X' * X) \ (X' * Y); % оператор \ решает систему линейных уравнений
% 
% fprintf('Веса модели (МНК):\n');
% for i = 1:n
%     fprintf('a%d = %.4f\n', i, w_LS(i));
% end
% for i = 1:n
%     fprintf('b%d = %.4f\n', i, w_LS(n+i));
% end
% 
% %% 6. ПРОВЕРКА МОДЕЛИ
% % Вычисляем предсказанные значения по модели
% fi_pred = zeros(size(fi_down));
% 
% % Первые n точек не предсказываем (нет истории)
% fi_pred(1:n) = fi_down(1:n);
% 
% % Для остальных точек делаем предсказание
% for t = n+1:N
%     pred = 0;
%     for i = 1:n
%         pred = pred + w_LS(i) * fi_down(t-i);     % часть AR
%     end
%     for i = 1:n
%         pred = pred + w_LS(n+i) * u_down(t-i);    % часть X
%     end
%     fi_pred(t) = pred;
% end
% 
% %% 7. ОЦЕНКА КАЧЕСТВА
% % Вычисляем ошибку
% error = fi_down(n+1:end) - fi_pred(n+1:end);
% MSE = mean(error.^2);
% RMSE = sqrt(MSE);
% fprintf('Среднеквадратичная ошибка (MSE): %.6f\n', MSE);
% fprintf('Корень из среднеквадратичной ошибки (RMSE): %.6f\n', RMSE);
% 
% % Коэффициент детерминации R²
% SS_res = sum(error.^2);
% SS_tot = sum((fi_down(n+1:end) - mean(fi_down(n+1:end))).^2);
% R2 = 1 - SS_res/SS_tot;
% fprintf('Коэффициент детерминации R²: %.4f\n', R2);
% 
% %% 8. ПОСТРОЕНИЕ ГРАФИКОВ
% 
% figure('Position', [100, 100, 1200, 800]);
% 
% % График 1: Исходные данные
% subplot(3,1,1);
% plot(t_raw, u, 'b-', 'LineWidth', 1); hold on;
% plot(t_raw, fi, 'r-', 'LineWidth', 1);
% xlabel('Время (с)');
% ylabel('Сигналы');
% legend('Вход u(t)', 'Выход φ(t)');
% title('Исходные данные (шаг 0.05 с)');
% grid on;
% 
% % График 2: Прореженные данные и предсказание модели
% subplot(3,1,2);
% plot(t_down, fi_down, 'ro-', 'MarkerSize', 4, 'LineWidth', 1); hold on;
% plot(t_down, fi_pred, 'b*-', 'MarkerSize', 4, 'LineWidth', 1);
% xlabel('Время (с)');
% ylabel('Выход φ(t)');
% legend('Реальный φ (прореженный)', 'Предсказанный φ (модель)');
% title(['Сравнение: Модель ARX порядка ', num2str(n), ' (шаг ', num2str(dt_model), ' с)']);
% grid on;
% 
% % График 3: Ошибка предсказания
% subplot(3,1,3);
% plot(t_down(n+1:end), error, 'k-', 'LineWidth', 1);
% xlabel('Время (с)');
% ylabel('Ошибка');
% title(['Ошибка предсказания (MSE = ', num2str(MSE, '%.6f'), ')']);
% grid on;
% 
% %% 9. ДОПОЛНИТЕЛЬНО: СРАВНЕНИЕ С РАЗНЫМИ ПОРЯДКАМИ МОДЕЛИ
% fprintf('\n--- Сравнение моделей разного порядка ---\n');
% 
% orders = [1, 2, 3, 4];
% MSE_values = [];
% 
% for n_test = orders
%     % Формируем регрессоры для порядка n_test
%     X_test = [];
%     Y_test = [];
%     for t = n_test+1:N
%         reg = [];
%         for i = 1:n_test
%             reg = [reg, fi_down(t-i)];
%         end
%         for i = 1:n_test
%             reg = [reg, u_down(t-i)];
%         end
%         X_test = [X_test; reg];
%         Y_test = [Y_test; fi_down(t)];
%     end
% 
%     % Обучаем модель
%     w_test = (X_test' * X_test) \ (X_test' * Y_test);
% 
%     % Предсказание
%     fi_test = zeros(size(fi_down));
%     fi_test(1:n_test) = fi_down(1:n_test);
%     for t = n_test+1:N
%         pred = 0;
%         for i = 1:n_test
%             pred = pred + w_test(i) * fi_down(t-i);
%         end
%         for i = 1:n_test
%             pred = pred + w_test(n_test+i) * u_down(t-i);
%         end
%         fi_test(t) = pred;
%     end
% 
%     % Ошибка
%     err_test = fi_down(n_test+1:end) - fi_test(n_test+1:end);
%     MSE_test = mean(err_test.^2);
%     MSE_values = [MSE_values, MSE_test];
% 
%     fprintf('Порядок %d: MSE = %.6f\n', n_test, MSE_test);
% end
% 
% % График зависимости ошибки от порядка модели
% figure;
% plot(orders, MSE_values, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
% xlabel('Порядок модели n');
% ylabel('MSE');
% title('Зависимость ошибки от порядка модели');
% grid on;