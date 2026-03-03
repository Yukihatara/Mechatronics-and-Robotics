clear; close all; clc

% --- Параметры системы ---
u = 1.0;      % Продольная скорость относительно воды, м/с
v = 0.3;      % Поперечная скорость относительно воды, м/с
u_curr = 0.9; % Скорость течения по X, м/с
v_curr = 0.2; % Скорость течения по Y, м/с
r = 0.1;      % Угловая скорость, рад/с
T = 30;      % Время моделирования, с

% --- Начальные условия ---
y0 = [0; 0; 0];

% --- Интервал времени ---
tspan = [0 T];

% --- Решение системы для двух случаев ---
% Случай 1: Без течения (u_curr = 0, v_curr = 0)
[t, y] = ode45(@(t, y) ship_ode(t, y, u, v, r, 0, 0), tspan, y0);

% Случай 2: С течением
[t1, y1] = ode45(@(t, y) ship_ode(t, y, u, v, r, u_curr, v_curr), tspan, y0);

% --- Извлечение результатов ---
x = y(:, 1);      % Без течения
y_pos = y(:, 2);
x1 = y1(:, 1);    % С течением
y1_pos = y1(:, 2);
psi = y1(:, 3);   % Угол (одинаков для обоих случаев)

% --- Построение графиков ---

% Траектории
plot(x, y_pos, 'b-', 'LineWidth', 2); hold on;
plot(x1, y1_pos, 'r-', 'LineWidth', 2);
xlabel('X, м');
ylabel('Y, м');
title('Траектория движения');
legend('Без течения', 'С течением', 'Location', 'best');
grid on; 
axis equal;
hold off;

% ----------------------------------------------------------------------- %
% === Параметры треугольника ===
side_length = 240;            % Длина стороны треугольника (м)
% Внутренний угол треугольника 60°, внешний угол поворота 120° (2.094 рад)
turn_angle = 120 * pi/180;   % Угол поворота в радианах

straight_time = side_length / u;  % Время движения по прямой в стоячей воде

% Время поворота на 120 градусов с угловой скоростью r_max
turn_time = turn_angle / r;

y0_part = [0; 0; 0]; % [X, Y, psi]
t_current = 0;

t_total = []; % Здесь соберем все времена
y_total = []; % Здесь соберем все решения

% Цикл по трем сторонам
for side = 1:3
    % --- 1. Движение по прямой (r = 0) ---
    t_span = [0, straight_time];
    [t_part, y_part] = ode45(@(t, y) ship_ode(t, y, u, v, 0, 0, 0), t_span, y0_part);
    % Сохраняем результаты
    t_total = [t_total; t_current + t_part];
    y_total = [y_total; y_part];
    % Обновляем текущее время и начальные условия для следующего участка
    t_current = t_current + straight_time;
    y0_part = y_part(end, :)'; % Последняя точка стала начальной для следующего этапа

    % --- 2. Поворот (кроме последней стороны, после которой не нужно поворачивать) ---
    if side < 3
        t_span = [0, turn_time];
        % Поворачиваем с положительной угловой скоростью
        [t_part, y_part] = ode45(@(t, y) ship_ode(t, y, u, v, r, 0, 0), t_span, y0_part);
        % Сохраняем результаты
        t_total = [t_total; t_current + t_part];
        y_total = [y_total; y_part];
        % Обновляем
        t_current = t_current + turn_time;
        y0_part = y_part(end, :)';
    end
end

% Аналогичный процесс для движения С ТЕЧЕНИЕМ
% Начальные условия
y0_part_curr = [0; 0; 0];
t_current_curr = 0;
t_total_curr = [];
y_total_curr = [];

for side = 1:3
    % --- 1. Движение по прямой (r = 0) С течением ---
    t_span = [0, straight_time];
    [t_part, y_part] = ode45(@(t, y) ship_ode(t, y, u, v, 0, u_curr, v_curr), t_span, y0_part_curr);
    t_total_curr = [t_total_curr; t_current_curr + t_part];
    y_total_curr = [y_total_curr; y_part];
    t_current_curr = t_current_curr + straight_time;
    y0_part_curr = y_part(end, :)';

    if side < 3
        t_span = [0, turn_time];
        % Поворот тоже происходит на течении!
        [t_part, y_part] = ode45(@(t, y) ship_ode(t, y, u, v, r, u_curr, v_curr), t_span, y0_part_curr);
        t_total_curr = [t_total_curr; t_current_curr + t_part];
        y_total_curr = [y_total_curr; y_part];
        t_current_curr = t_current_curr + turn_time;
        y0_part_curr = y_part(end, :)';
    end
end

% === Визуализация ===
% figure('Position', [100, 100, 1000, 800]);

% График траекторий
subplot(2,2,[1,2]);
plot(y_total(:,1), y_total(:,2), 'b-', 'LineWidth', 2); hold on;
plot(y_total_curr(:,1), y_total_curr(:,2), 'r-', 'LineWidth', 2);
plot(y_total(1,1), y_total(1,2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % Старт
plot(y_total(end,1), y_total(end,2), 'rx', 'MarkerSize', 10, 'LineWidth', 2); % Финиш без течения
plot(y_total_curr(end,1), y_total_curr(end,2), 'ks', 'MarkerSize', 10, 'LineWidth', 2); % Финиш с течением
xlabel('X, м');
ylabel('Y, м');
title('Движение ПА по треугольнику');
legend('Без течения', 'С течением', 'Старт', 'Финиш (без теч.)', 'Финиш (с теч.)', 'Location','best');
grid on; axis equal;
hold off;

% График угла рыскания
subplot(2,2,3);
plot(t_total, y_total(:,3)*180/pi, 'b-', 'LineWidth', 1.5); hold on;
plot(t_total_curr, y_total_curr(:,3)*180/pi, 'r-', 'LineWidth', 1.5);
xlabel('Время, с');
ylabel('\psi, град');
title('Угол рыскания');
legend('Без течения', 'С течением');
grid on;

% График путевого угла (для информации)
subplot(2,2,4);
% Вычисляем путевой угол как atan2(dY, dX)
dX = gradient(y_total(:,1), t_total);
dY = gradient(y_total(:,2), t_total);
course_angle = atan2(dY, dX)*180/pi;
dX_curr = gradient(y_total_curr(:,1), t_total_curr);
dY_curr = gradient(y_total_curr(:,2), t_total_curr);
course_angle_curr = atan2(dY_curr, dX_curr)*180/pi;
plot(t_total, course_angle, 'b-', 'LineWidth', 1.5); hold on;
plot(t_total_curr, course_angle_curr, 'r-', 'LineWidth', 1.5);
xlabel('Время, с');
ylabel('Курс, град');
title('Путевой угол (направление движения)');
legend('Без течения', 'С течением');
grid on;






% --- Определение функции, описывающей систему ---
function dydt = ship_ode(~, y, u, v, r, u_curr, v_curr)
    % y(1) = x, y(2) = y, y(3) = psi
    
    psi_cur = y(3); % Текущий угол
    
    % Скорость относительно воды + глобальное течение
    dxdt = cos(psi_cur) * u - sin(psi_cur) * v + u_curr;
    dydt_sys = sin(psi_cur) * u + cos(psi_cur) * v + v_curr;
    dpsidt = r;
    
    % Выходной вектор-столбец
    dydt = [dxdt; dydt_sys; dpsidt];
end