%% 1.1 Логическое И (AND)

%% Задание 1: Персептрон для функций И и ИЛИ
clear; clc; close all;

% 1.1 Логическое И (AND)
fprintf('========== ФУНКЦИЯ И (AND) ==========\n');

% Входы и цели
P_and = [0 0 1 1;
         0 1 0 1];
T_and = [0 0 0 1];

% Создание и обучение
net_and = newp([0 1; 0 1], 1);
net_and.trainParam.epochs = 10;
net_and = train(net_and, P_and, T_and);

% Проверка
Y_and = sim(net_and, P_and);

% Вывод результатов
fprintf('\n--- РЕЗУЛЬТАТЫ ---\n');
fprintf('На вход подаем: x1 и x2 (0 или 1)\n');
fprintf('На выходе получаем: 0 или 1 (результат операции И)\n\n');
fprintf(' x1  x2 | Ожидается | Получено\n');
fprintf('-----------------------------\n');
for i = 1:4
    fprintf(' %d   %d  |     %d      |    %d\n', P_and(1,i), P_and(2,i), T_and(i), Y_and(i));
end

% Веса и разделяющая линия
w1_and = net_and.IW{1}(1);
w2_and = net_and.IW{1}(2);
b_and = net_and.b{1};
fprintf('\nРазделяющая линия: %.1f*x1 + %.1f*x2 + %.1f = 0\n', w1_and, w2_and, b_and);
fprintf('(Если значение > 0 -> класс 1, если < 0 -> класс 0)\n\n');

% 1.2 Логическое ИЛИ (OR)
fprintf('\n========== ФУНКЦИЯ ИЛИ (OR) ==========\n');

P_or = [0 0 1 1;
        0 1 0 1];
T_or = [0 1 1 1];

net_or = newp([0 1; 0 1], 1);
net_or.trainParam.epochs = 10;
net_or = train(net_or, P_or, T_or);

Y_or = sim(net_or, P_or);

fprintf('\n--- РЕЗУЛЬТАТЫ ---\n');
fprintf('На вход подаем: x1 и x2 (0 или 1)\n');
fprintf('На выходе получаем: 0 или 1 (результат операции ИЛИ)\n\n');
fprintf(' x1  x2 | Ожидается | Получено\n');
fprintf('-----------------------------\n');
for i = 1:4
    fprintf(' %d   %d  |     %d      |    %d\n', P_or(1,i), P_or(2,i), T_or(i), Y_or(i));
end

% Веса и разделяющая линия
w1_or = net_or.IW{1}(1);
w2_or = net_or.IW{1}(2);
b_or = net_or.b{1};
fprintf('\nРазделяющая линия: %.1f*x1 + %.1f*x2 + %.1f = 0\n', w1_or, w2_or, b_or);
fprintf('(Если значение > 0 -> класс 1, если < 0 -> класс 0)\n');

% 2. Визуализация с разделяющими линиями
figure;

% Подграфик для И
subplot(1,2,1);
hold on; grid on;
for i = 1:4
    if T_and(i) == 1
        plot(P_and(1,i), P_and(2,i), 'ro', 'MarkerSize', 15, 'LineWidth', 2);
    else
        plot(P_and(1,i), P_and(2,i), 'bs', 'MarkerSize', 15, 'LineWidth', 2);
    end
end
% Разделяющая линия для И
x1 = linspace(-0.5, 1.5, 100);
x2 = -(w1_and*x1 + b_and)/w2_and;
plot(x1, x2, 'g-', 'LineWidth', 2);
xlabel('x1'); ylabel('x2');
title('Функция И (AND)');
legend('Класс 1', 'Класс 0', 'Разделяющая линия', 'Location', 'best');
xlim([-0.2 1.2]); ylim([-0.2 1.2]);
hold off;

% Подграфик для ИЛИ
subplot(1,2,2);
hold on; grid on;
for i = 1:4
    if T_or(i) == 1
        plot(P_or(1,i), P_or(2,i), 'ro', 'MarkerSize', 15, 'LineWidth', 2);
    else
        plot(P_or(1,i), P_or(2,i), 'bs', 'MarkerSize', 15, 'LineWidth', 2);
    end
end
% Разделяющая линия для ИЛИ
x2 = -(w1_or*x1 + b_or)/w2_or;
plot(x1, x2, 'g-', 'LineWidth', 2);
xlabel('x1'); ylabel('x2');
title('Функция ИЛИ (OR)');
legend('Класс 1', 'Класс 0', 'Разделяющая линия', 'Location', 'best');
xlim([-0.2 1.2]); ylim([-0.2 1.2]);
hold off;

%% Задание 2: Классификация с заданными векторами
clear; clc; close all;

% 1. Подготовка данных
% Обучающее множество
P = [-0.5 -0.5 0.3 -0.1;
     -0.5 0.5 -0.5 1.0];
T = [1 1 0 0];  % цели

% Тестирующее множество
p_test = [0.7; 1.2];

% 2. Создание и обучение персептрона
% Определяем диапазоны входов (мин и макс по каждой строке)
minmax = [min(P(1,:)) max(P(1,:));
          min(P(2,:)) max(P(2,:))];
fprintf('Диапазоны входов:\n');
fprintf('x1: от %.1f до %.1f\n', minmax(1,1), minmax(1,2));
fprintf('x2: от %.1f до %.1f\n', minmax(2,1), minmax(2,2));

% Создание персептрона
net = newp(minmax, 1);  % 1 выход
net.trainParam.epochs = 20;  % максимум 20 эпох
net.trainParam.show = 5;     % показывать прогресс каждые 5 эпох

% Обучение
fprintf('\nОбучение персептрона...\n');
net = train(net, P, T);

% 3. Проверка на обучающем множестве
Y_train = sim(net, P);
fprintf('\nРезультаты на обучающем множестве:\n');
for i = 1:4
    fprintf('Пример %d: (%.1f, %.1f) -> выход: %d (ожидалось: %d) %s\n', ...
            i, P(1,i), P(2,i), Y_train(i), T(i), ...
            chk(Y_train(i), T(i)));
end

% 4. Тестирование на новом векторе
Y_test = sim(net, p_test);
fprintf('\n--- Тестирование ---\n');
fprintf('Тестовый вектор: (%.1f, %.1f)\n', p_test(1), p_test(2));
fprintf('Класс: %d\n', Y_test);

% 5. Полученные веса и смещение
fprintf('\n--- Параметры обученной сети ---\n');
fprintf('Веса: w1 = %.4f, w2 = %.4f\n', net.IW{1}(1), net.IW{1}(2));
fprintf('Смещение (bias): %.4f\n', net.b{1});
fprintf('Разделяющая линия: %.4f*x1 + %.4f*x2 + %.4f = 0\n', ...
        net.IW{1}(1), net.IW{1}(2), net.b{1});

% 6. Визуализация
figure;
hold on;
grid on;

% Обучающие примеры (класс 1 - красные круги, класс 0 - синие квадраты)
for i = 1:4
    if T(i) == 1
        plot(P(1,i), P(2,i), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    else
        plot(P(1,i), P(2,i), 'bs', 'MarkerSize', 10, 'LineWidth', 2);
    end
end

% Тестовый вектор
plot(p_test(1), p_test(2), 'md', 'MarkerSize', 12, 'LineWidth', 2, ...
     'MarkerFaceColor', 'm');

% Разделяющая линия
w1 = net.IW{1}(1);
w2 = net.IW{1}(2);
b = net.b{1};

% Для построения линии берем x1 в диапазоне от -1 до 1.5
x1_line = linspace(-1, 1.5, 100);
% Из уравнения w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b)/w2
if abs(w2) > 1e-6  % если w2 не близок к нулю
    x2_line = -(w1*x1_line + b)/w2;
    plot(x1_line, x2_line, 'g-', 'LineWidth', 2);
else
    % Особый случай: если w2 ≈ 0, линия вертикальна x1 = -b/w1
    x1_vert = -b/w1;
    line([x1_vert x1_vert], [-1 2], 'Color', 'g', 'LineWidth', 2);
end

% Оформление
xlabel('x_1');
ylabel('x_2');
title('Классификация персептроном');
xlim([-1 1.5]);
ylim([-1 2]);
hold off;

% Вспомогательная функция для проверки
function s = chk(a, b)
    if a == b
        s = 'True';
    else
        s = 'False';
    end
end

%% Задание 3: Демонстрация XOR
clear; clc; close all;

% 1. Данные
P_xor = [0 0 1 1;
         0 1 0 1];
T_xor = [0 1 1 0];

% 2. Обучение
net_xor = newp([0 1; 0 1], 1);
net_xor.trainParam.epochs = 20;
net_xor = train(net_xor, P_xor, T_xor);

% 3. Результаты
Y_xor = sim(net_xor, P_xor);
fprintf('Ожидалось: %d %d %d %d\n', T_xor);
fprintf('Получено:  %d %d %d %d\n', Y_xor);

% 4. Визуализация
figure;
hold on; grid on;

% Точки
for i = 1:4
    if T_xor(i) == 1
        plot(P_xor(1,i), P_xor(2,i), 'ro', 'MarkerSize', 15, 'LineWidth', 2);
    else
        plot(P_xor(1,i), P_xor(2,i), 'bs', 'MarkerSize', 15, 'LineWidth', 2);
    end
end

% Разделяющая линия - упрощенный вариант, всегда рисуем
w1 = net_xor.IW{1}(1);
w2 = net_xor.IW{1}(2);
b = net_xor.b{1};

fprintf('\nВеса: w1 = %.3f, w2 = %.3f, b = %.3f\n', w1, w2, b);

x1_line = linspace(-0.5, 1.5, 100);

if abs(w2) > 0.001
    % Нормальный случай - линия по x2
    x2_line = -(w1*x1_line + b)/w2;
    plot(x1_line, x2_line, 'g-', 'LineWidth', 2);
elseif abs(w1) > 0.001
    % Если w2 близок к 0 - рисуем вертикальную линию
    x1_vert = -b/w1;
    plot([x1_vert x1_vert], [-0.5 1.5], 'g-', 'LineWidth', 2);
else
    % Если оба веса близки к 0 - просто горизонтальная линия
    plot(x1_line, zeros(size(x1_line)), 'g-', 'LineWidth', 2);
end

xlabel('x_1'); ylabel('x_2');
title('XOR: однослойный персептрон');
legend('Класс 1', 'Класс 0', 'Разделяющая линия', 'Location', 'best');
xlim([-0.5 1.5]); ylim([-0.5 1.5]);
hold off;


%% Задание 5: Распознавание бракованных кирпичей
clear; clc; close all;

% 1. Ввод данных из таблицы
% Данные: [частота 475 Гц, частота 557 Гц, качество]
% yes = 1 (хороший), no = 0 (брак)

% Обучающие данные (10 кирпичей)
P = [0.958 1.043 1.907 0.780 0.579 0.003 0.001 0.014 0.007 0.004;
     0.003 0.001 0.003 0.002 0.001 0.105 1.748 1.839 1.021 0.214];
T = [1 1 1 1 1 0 0 0 0 0]; % 1 - хороший, 0 - бракованный

fprintf('=== Распознавание бракованных кирпичей ===\n');
fprintf('Всего примеров: %d (5 хороших, 5 бракованных)\n\n', length(T));

% 2. Создание и обучение персептрона
% Определяем диапазоны входов
minmax = [min(P(1,:)) max(P(1,:));
          min(P(2,:)) max(P(2,:))];

% Создаем персептрон
net = newp(minmax, 1);
net.trainParam.epochs = 100;

% Обучаем
fprintf('Обучение персептрона...\n');
net = train(net, P, T);

% 3. Проверка на обучающих данных
Y = sim(net, P);

fprintf('\nРезультаты классификации:\n');
fprintf('----------------------------------------\n');
fprintf(' 475 Гц | 557 Гц | Факт | Распознано | Статус\n');
fprintf('----------------------------------------\n');

for i = 1:10
    if Y(i) == T(i)
        status = 'True';
    else
        status = 'False';
    end
    fprintf('%6.3f  | %6.3f |  %d   |     %d      |   %s\n', ...
            P(1,i), P(2,i), T(i), Y(i), status);
end
fprintf('----------------------------------------\n');

% Считаем ошибки
errors = sum(Y ~= T);
fprintf('\nОшибок: %d из %d (%.0f%%)\n', errors, length(T), errors/length(T)*100);

% 4. Полученные веса и разделяющая линия
w1 = net.IW{1}(1);
w2 = net.IW{1}(2);
b = net.b{1};

fprintf('\nПараметры разделяющей линии:\n');
fprintf('w1 = %.3f, w2 = %.3f, b = %.3f\n', w1, w2, b);
fprintf('Уравнение: %.3f*x1 + %.3f*x2 + %.3f = 0\n', w1, w2, b);

% 5. Визуализация
figure;
hold on; grid on;

% Хорошие кирпичи (класс 1) - зеленые круги
good_idx = find(T == 1);
plot(P(1, good_idx), P(2, good_idx), 'go', 'MarkerSize', 10, 'LineWidth', 2, ...
     'MarkerFaceColor', 'g');

% Бракованные кирпичи (класс 0) - красные квадраты
bad_idx = find(T == 0);
plot(P(1, bad_idx), P(2, bad_idx), 'rs', 'MarkerSize', 10, 'LineWidth', 2, ...
     'MarkerFaceColor', 'r');

% Разделяющая линия
x1_line = linspace(0, 2, 100);
if abs(w2) > 1e-6
    x2_line = -(w1*x1_line + b)/w2;
    plot(x1_line, x2_line, 'b-', 'LineWidth', 2);
end

xlabel('Интенсивность на 475 Гц');
ylabel('Интенсивность на 557 Гц');
title('Классификация кирпичей');
legend('Хорошие', 'Бракованные', 'Разделяющая линия', 'Location', 'best');
xlim([0 2]); ylim([0 2]);
hold off;

% 6. Проверка для нового кирпича (пример)
% Возьмем средние значения для демонстрации
p_new = [0.5; 0.5];
Y_new = sim(net, p_new);
fprintf('\n--- Проверка нового кирпича ---\n');
fprintf('Вход: (%.1f Гц, %.1f Гц) -> ', p_new(1), p_new(2));
if Y_new == 1
    fprintf('КИРПИЧ ХОРОШИЙ\n');
else
    fprintf('КИРПИЧ БРАКОВАННЫЙ\n');
end