function horizon_hough(input_image_path)
    %% Загрузка и подготовка изображения
    img = imread(input_image_path);
    gray_img = rgb2gray(img);

    %% 1. Детектор границ Канни
    sigma = 7;  % сила размытия (можно регулировать)
    gray_img_blurred = imgaussfilt(gray_img, sigma);
    bw = edge(gray_img_blurred, 'canny');

    %% 2. Преобразование Хафа с ограничением углов
    % Допустимый диапазон: [-90, 89]
    % В методичке: [-90:-60] ∪ [60:90], но 90 заменяем на 89
    theta_range = [-90:1:-80, 80:1:89];

    [H, theta, rho] = hough(bw, 'Theta', theta_range);

    %% 3. Поиск пиков
    peaks = houghpeaks(H, 10, 'Threshold', 0.3 * max(H(:)));

    %% 4. Извлечение линий
    % FillGap: 2.5% от ширины изображения
    fill_gap_value = round(0.015 * size(img, 2));
    % MinLength: 20% от высоты изображения (эмпирически)
    min_length_value = round(0.2 * size(img, 1));

    lines = houghlines(bw, theta, rho, peaks, ...
                       'FillGap', fill_gap_value, ...
                       'MinLength', min_length_value);

    %% 5. Поиск самой длинной линии
    if isempty(lines)
        error('Горизонт не найден. Попробуйте изменить параметры Canny или углы.');
    end

    max_len = 0;
    best_line = lines(1);

    for k = 1:length(lines)
        len = norm(lines(k).point1 - lines(k).point2);
        if len > max_len
            max_len = len;
            best_line = lines(k);
        end
    end

%% 5. Поиск самой длинной линии
if isempty(lines)
    error('Горизонт не найден. Попробуйте изменить параметры Canny или углы.');
end

max_len = 0;
best_line = lines(1);
best_line_idx = 0;

for k = 1:length(lines)
    len = norm(lines(k).point1 - lines(k).point2);
    if len > max_len
        max_len = len;
        best_line = lines(k);
        best_line_idx = k;
    end
end

%% 6. Отображение результата
figure('Name', 'Обнаружение горизонта (Хаф)', 'NumberTitle', 'off');
imshow(img); hold on;

% Жёлтая линия горизонта
xy = [best_line.point1; best_line.point2];

% Растягиваем координаты x, чтобы неполная линия стала полной
xy(:,1) = [0,size(img, 2)]

plot(xy(:,1), xy(:,2), 'y-', 'LineWidth', 3);

title('Линия горизонта (преобразование Хафа)', 'FontSize', 12);

for i = 1:length(lines)
    xy1 = [lines(i).point1; lines(i).point2];
    plot(xy1(:,1), xy1(:,2), 'r-', 'LineWidth', 3);
end

    %% 7. Вывод параметров
    fprintf('\n========== РЕЗУЛЬТАТ (ХАФ) ==========\n');
    fprintf('Файл: %s\n', input_image_path);
    fprintf('Угол (theta): %.2f градусов\n', best_line.theta);
    fprintf('Расстояние (rho): %.2f пикселей\n', best_line.rho);
    fprintf('Длина линии: %.2f пикселей\n', max_len);
    fprintf('Точки: (%d, %d) -> (%d, %d)\n', ...
            best_line.point1(1), best_line.point1(2), ...
            best_line.point2(1), best_line.point2(2));
    fprintf('=====================================\n\n');

    %% 8. Дополнительно: покажем промежуточные этапы
    figure('Name', 'Этапы обработки', 'NumberTitle', 'off');

    subplot(2,2,1);
    imshow(gray_img);
    title('Исходное (grayscale)');

    subplot(2,2,2);
    imshow(bw);
    title('Границы Canny');

    subplot(2,2,3);
    imshow(gray_img_blurred);
    title('Исходное (grayscale)');    

    subplot(2,2,4);
    imshow(H, [], 'XData', theta, 'YData', rho, 'InitialMagnification', 'fit');
    axis on, axis normal;
    xlabel('\theta (градусы)'), ylabel('\rho (пиксели)');
    title('Пространство Хафа');
    colormap(gca, hot);
end

function horizon_kmeans_intense(input_image_path)
    %% Загрузка и подготовка изображения
    img = imread(input_image_path);
    gray_img = rgb2gray(img);
    [h, w] = size(gray_img);
    
    %% 1. ГАУССОВСКИЙ ФИЛЬТР (предварительное сглаживание)
    sigma = 3;  % сила размытия (подбери под своё изображение)
    gray_smoothed = imgaussfilt(gray_img, sigma);
    
    %% Подготовка данных - только яркость
    features = double(gray_smoothed(:));
    
    %% K-means кластеризация
    rng(42);
    [cluster_idx, cluster_centers] = kmeans(features, 2, ...
                                             'Replicates', 5, ...
                                             'Display', 'off');
    
    %% Определяем небо и море по яркости
    if cluster_centers(1) > cluster_centers(2)
        sky_idx = 1;
        sea_idx = 2;
    else
        sky_idx = 2;
        sea_idx = 1;
    end
    
    %% Восстанавливаем карту кластеров
    cluster_img = reshape(cluster_idx, h, w);
    
    %% 2. МОРФОЛОГИЧЕСКАЯ ОБРАБОТКА маски
    sky_mask = (cluster_img == sky_idx);
    
    % Создаём structuring element (диск радиусом 5 пикселей)
    se = strel('disk', 50);
    
    % Применяем морфологические операции:
    % - imopen: эрозия + дилатация (убирает мелкие шумы - "соль")
    % - imclose: дилатация + эрозия (закрывает дыры - "перец")
    
    % Сначала открытие, чтобы убрать мелкие островки неба в море
    sky_mask = imopen(sky_mask, se);
    
    % Потом закрытие, чтобы убрать дыры в небе
    sky_mask = imclose(sky_mask, se);
    
    % Дополнительно: заполняем маленькие дыры (опционально)
    sky_mask = bwareaopen(sky_mask, 100);  % убираем объекты меньше 100 пикселей
    
    %% 3. НАХОЖДЕНИЕ ГРАНИЦЫ с улучшенным алгоритмом
    horizon_y = zeros(1, w);
    
    for col = 1:w
        % Берём столбец
        column = sky_mask(:, col);
        
        % Ищем переход сверху вниз
        % Находим все переходы из 1 в 0
        transitions = find(diff([1; column; 0]) == -1);
        
        if isempty(transitions)
            % Если нет перехода - вся колонка однородна
            if column(1) == 1  % всё небо
                horizon_y(col) = h;
            else  % всё море
                horizon_y(col) = 1;
            end
        else
            % Берём ПЕРВЫЙ переход (самый верхний)
            % Это даст верхнюю границу неба
            horizon_y(col) = transitions(1);
            
            % Альтернатива: можно брать ПОСЛЕДНИЙ переход (нижняя граница неба)
            % horizon_y(col) = transitions(end);
        end
    end
    
    %% 4. ДОПОЛНИТЕЛЬНАЯ ФИЛЬТРАЦИЯ профиля
    % Медианная фильтрация (уже была)
    horizon_y = medfilt1(horizon_y, 5);
    
    % Отбрасываем выбросы (заменяем на соседние)
    % Находим точки, сильно отличающиеся от соседей
    for iter = 1:3  % несколько итераций
        for col = 2:w-1
            if abs(horizon_y(col) - horizon_y(col-1)) > 50 && ...
               abs(horizon_y(col) - horizon_y(col+1)) > 50
                horizon_y(col) = round((horizon_y(col-1) + horizon_y(col+1)) / 2);
            end
        end
    end
    
    %% 5. АППРОКСИМАЦИЯ ПРЯМОЙ (с отбрасыванием выбросов)
    x = 1:w;
    
    % Опционально: отбрасываем явные выбросы для аппроксимации
    % Используем только точки, где горизонт не у границ
    valid_idx = (horizon_y > 10) & (horizon_y < h-10);
    
    if sum(valid_idx) > w/2  % если достаточно хороших точек
        p = polyfit(x(valid_idx), horizon_y(valid_idx), 1);
    else
        p = polyfit(x, horizon_y, 1);  % иначе используем все
    end
    
    y_fit = polyval(p, [1, w]);
    
    %% ВИЗУАЛИЗАЦИЯ
    figure('Name', 'Горизонт (K-means + морфология)', 'NumberTitle', 'off', ...
           'Position', [100 100 1600 1000]);
    
    % 1. Исходное с горизонтом
    subplot(3,4,1);
    imshow(img); hold on;
    plot([1, w], y_fit, 'y-', 'LineWidth', 3);
    title('Итоговый горизонт');
    
    % 2. Кластеризация (сырая)
    subplot(3,4,2);
    imshow(label2rgb(cluster_img));
    title('Сырая кластеризация');
    
    % 3. Маска неба (сырая)
    subplot(3,4,3);
    raw_mask = (cluster_img == sky_idx);
    imshow(raw_mask);
    title('Сырая маска неба');
    
    % 4. Маска после морфологии
    subplot(3,4,4);
    imshow(sky_mask);
    title('Маска после морфологии');
    
    % 5. Профиль горизонта (все точки)
    subplot(3,4,[5,6]);
    plot(x, horizon_y, 'b.-', 'MarkerSize', 3); hold on;
    plot(x(valid_idx), horizon_y(valid_idx), 'g.', 'MarkerSize', 5);
    plot([1, w], y_fit, 'r-', 'LineWidth', 2);
    xlabel('Колонка'); ylabel('Строка');
    title('Профиль горизонта');
    legend({'Все точки', 'Использованные', 'Аппроксимация'});
    grid on;
    ylim([1 h]);
    
    % 7. Гистограмма яркости
    subplot(3,4,[7 8]);
    histogram(gray_img(:), 50, 'FaceAlpha', 0.5, 'EdgeColor', 'none'); hold on;
    histogram(gray_smoothed(:), 50, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    xline(cluster_centers(sky_idx), 'b-', 'LineWidth', 2);
    xline(cluster_centers(sea_idx), 'r-', 'LineWidth', 2);
    title('Гистограмма яркости');
    legend({'Исходная', 'Сглаженная', 'Небо', 'Море'});
     
    % 9-12. Попиксельный горизонт на изображении
    subplot(3,4,9);
    imshow(img); hold on;
    plot(horizon_y, 'b.', 'MarkerSize', 1);
    title('Попиксельный горизонт');
    
    subplot(3,4,10);
    imshow(img); hold on;
    plot(horizon_y, 'b.', 'MarkerSize', 1);
    plot([1, w], y_fit, 'y-', 'LineWidth', 2);
    title('Попиксельный + аппроксимация');
    
    subplot(3,4,11);
    imshow(img); hold on;
    sky_outline = bwperim(sky_mask);
    [r, c] = find(sky_outline);
    plot(c, r, 'c.', 'MarkerSize', 1);
    title('Контур маски неба');
    
    subplot(3,4,12);
    imshow(img); hold on;
    % Наложение полупрозрачной маски
    h_im = imshow(label2rgb(sky_mask));
    set(h_im, 'AlphaData', 0.3);
    plot([1, w], y_fit, 'y-', 'LineWidth', 3);
    title('Наложение маски');
    
    %% ВЫВОД ПАРАМЕТРОВ
    fprintf('\n========== РЕЗУЛЬТАТ (С МОРФОЛОГИЕЙ) ==========\n');
    fprintf('Параметры Гаусса: sigma = %d\n', sigma);
    fprintf('Яркость неба: %.1f\n', cluster_centers(sky_idx));
    fprintf('Яркость моря: %.1f\n', cluster_centers(sea_idx));
    fprintf('Уравнение горизонта: y = %.2f * x + %.2f\n', p(1), p(2));
    fprintf('Качество аппроксимации: использовано %d из %d точек (%.1f%%)\n', ...
            sum(valid_idx), w, 100*sum(valid_idx)/w);
    fprintf('===============================================\n\n');


    %% ===== ДОПОЛНЕНИЕ: BOUNDING BOXES И ЦЕНТРЫ =====
    % Создаём отдельную фигуру
    %% ===== ДОПОЛНЕНИЕ: BOUNDING BOXES, ЦЕНТРЫ И ПЕРПЕНДИКУЛЯР =====
    % Создаём отдельную фигуру
    figure('Name', 'Центры кластеров и линия горизонта', 'NumberTitle', 'off', ...
           'Position', [200 200 800 600]);
    
    imshow(img); hold on;
    
    % Находим bounding boxes для неба и моря
    sky_stats = regionprops(sky_mask, 'BoundingBox', 'Centroid', 'Area');
    sea_stats = regionprops(~sky_mask, 'BoundingBox', 'Centroid', 'Area');
    
    if ~isempty(sky_stats)
        % Берём самую большую компоненту неба
        [~, idx] = max([sky_stats.Area]);
        sky_bbox = sky_stats(idx).BoundingBox;
        sky_center = sky_stats(idx).Centroid;
        
        % Рисуем bounding box и центр неба
        rectangle('Position', sky_bbox, 'EdgeColor', 'b', 'LineWidth', 1.5, 'LineStyle', '--');
        plot(sky_center(1), sky_center(2), 'b*', 'MarkerSize', 15, 'LineWidth', 2);
        text(sky_center(1)+10, sky_center(2)-10, 'Небо', 'Color', 'b', 'FontSize', 12);
    end
    
    if ~isempty(sea_stats)
        % Берём самую большую компоненту моря
        [~, idx] = max([sea_stats.Area]);
        sea_bbox = sea_stats(idx).BoundingBox;
        sea_center = sea_stats(idx).Centroid;
        
        % Рисуем bounding box и центр моря
        rectangle('Position', sea_bbox, 'EdgeColor', 'r', 'LineWidth', 1.5, 'LineStyle', '--');
        plot(sea_center(1), sea_center(2), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
        text(sea_center(1)+10, sea_center(2)+10, 'Море', 'Color', 'r', 'FontSize', 12);
    end
    
    % Если нашли оба центра - строим геометрию
    if exist('sky_center', 'var') && exist('sea_center', 'var')
        % Линия, соединяющая центры
        plot([sky_center(1), sea_center(1)], [sky_center(2), sea_center(2)], ...
             'g-', 'LineWidth', 2);
        
        % Вектор от моря к небу
        vec = [sky_center(1) - sea_center(1), sky_center(2) - sea_center(2)];
        
        % Нормализуем
        if norm(vec) > 0
            vec = vec / norm(vec);
        end
        
        % Перпендикулярный вектор
        perp_vec = [-vec(2), vec(1)];
        
        % Середина между центрами
        mid_point = [(sky_center(1) + sea_center(1))/2, (sky_center(2) + sea_center(2))/2];
        
        % Рисуем середину
        plot(mid_point(1), mid_point(2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'y');
        
        % Строим перпендикуляр через середину (длиной в 200 пикселей в обе стороны)
        line_length = 200;
        perp_x = [mid_point(1) - perp_vec(1)*line_length, mid_point(1) + perp_vec(1)*line_length];
        perp_y = [mid_point(2) - perp_vec(2)*line_length, mid_point(2) + perp_vec(2)*line_length];
        
        % Рисуем перпендикуляр (горизонт)
        plot(perp_x, perp_y, 'y-', 'LineWidth', 3);
        
        % Подписи
        text(mid_point(1)+10, mid_point(2)-10, 'Середина', 'Color', 'k', 'FontSize', 10);
        text(mid_point(1)+50, mid_point(2)-30, 'ГОРИЗОНТ', 'Color', 'y', 'FontSize', 12, 'FontWeight', 'bold');
    end

title('Геометрия: центры, линия центров и перпендикуляр (горизонт)', 'FontSize', 14);
hold off;


end

function horizon_threshold(input_image_path)
    %% Загрузка и подготовка изображения
    img = imread(input_image_path);
    gray_img = rgb2gray(img);
    [h, w] = size(gray_img);
    
    %% 1. ГАУССОВСКИЙ ФИЛЬТР (сглаживание)
    sigma = 2;
    gray_smoothed = imgaussfilt(gray_img, sigma);
    
    %% 2. ОПРЕДЕЛЕНИЕ ПОРОГА (метод Otsu)
    threshold = graythresh(gray_smoothed);  % значение в [0, 1]
    threshold_val = threshold * 255;  % переводим в яркость 0-255
    
    %% 3. БИНАРИЗАЦИЯ
    % Небо = ярче порога (1), море = темнее порога (0)
    bw = gray_smoothed > threshold_val;
    
    % Проверяем, что небо действительно вверху
    top_third = mean(bw(1:floor(h/3), :), 'all');
    bottom_third = mean(bw(floor(2*h/3):end, :), 'all');
    
    if top_third < bottom_third
        % Если вверху больше моря (0), чем неба (1) - инвертируем
        bw = ~bw;
        fprintf('Маска инвертирована: небо оказалось темнее моря\n');
    end
    
    %% 4. МОРФОЛОГИЧЕСКАЯ ОЧИСТКА
    % Убираем мелкий шум
    se = strel('disk', 5);
    bw = imopen(bw, se);
    bw = imclose(bw, se);
    bw = bwareaopen(bw, 500);  % убираем объекты меньше 500 пикселей
    bw = imfill(bw, 'holes');
    
    %% 5. НАХОЖДЕНИЕ ЛИНИИ ГОРИЗОНТА
    % Для каждой колонки находим границу между небом и морем
    horizon_y = zeros(1, w);
    
    for col = 1:w
        column = bw(:, col);
        
        % Ищем первый переход сверху вниз из 1 (небо) в 0 (море)
        % Добавляем 1 в начало и 0 в конец для корректного поиска
        transitions = find(diff([1; column; 0]) == -1);
        
        if isempty(transitions)
            % Нет перехода - вся колонка однородна
            if column(1) == 1  % всё небо
                horizon_y(col) = h;  % горизонт внизу
            else  % всё море
                horizon_y(col) = 1;   % горизонт вверху
            end
        else
            % Берём первый (самый верхний) переход
            horizon_y(col) = transitions(1);
        end
    end
    
    %% 6. СГЛАЖИВАНИЕ ЛИНИИ ГОРИЗОНТА
    % Медианная фильтрация для удаления выбросов
    horizon_y = medfilt1(horizon_y, 7);
    
    %% 7. АППРОКСИМАЦИЯ ПРЯМОЙ (линия горизонта)
    x = 1:w;
    
    % Отбрасываем явные выбросы (слишком близко к краям)
    valid_idx = (horizon_y > 5) & (horizon_y < h-5);
    
    if sum(valid_idx) > w/3
        % Аппроксимируем прямой линией
        p = polyfit(x(valid_idx), horizon_y(valid_idx), 1);
    else
        % Если слишком мало хороших точек - берём медиану
        p = [0, median(horizon_y)];
    end
    
    % Вычисляем координаты прямой на границах изображения
    y_left = polyval(p, 1);
    y_right = polyval(p, w);
    
    %% 8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТА
    figure('Name', 'Пороговая сегментация - линия горизонта', 'NumberTitle', 'off', ...
           'Position', [100 100 1400 900]);
    
    % 1. Исходное изображение с линией горизонта
    subplot(2,3,1);
    imshow(img); hold on;
    plot([1, w], [y_left, y_right], 'y-', 'LineWidth', 2);
    title('Линия горизонта (пороговая обработка)');
    
    % 2. Бинарное изображение
    subplot(2,3,2);
    imshow(bw);
    title('Бинаризация (небо=белое)');
    
    % 3. Гистограмма яркости с порогом
    subplot(2,3,3);
    histogram(gray_img(:), 50, 'FaceColor', [0.7 0.7 0.7]); hold on;
    histogram(gray_smoothed(:), 50, 'FaceColor', [0.3 0.3 0.3], 'FaceAlpha', 0.7);
    xline(threshold_val, 'r-', 'LineWidth', 2);
    xlabel('Яркость'); ylabel('Количество');
    title('Гистограмма яркости');
    legend({'Исходная', 'Сглаженная', 'Порог'});
    
    % 4. Профиль горизонта (попиксельно)
    subplot(2,3,4);
    plot(x, horizon_y, 'b.-', 'MarkerSize', 3); hold on;
    plot(x(valid_idx), horizon_y(valid_idx), 'g.', 'MarkerSize', 8);
    plot([1, w], [y_left, y_right], 'r-', 'LineWidth', 2);
    xlabel('Колонка'); ylabel('Строка');
    title('Профиль линии горизонта');
    legend({'Все точки', 'Использованные', 'Аппроксимация'});
    grid on;
    ylim([1 h]);
    
    % 5. Наложение маски
    subplot(2,3,5);
    imshow(img); hold on;
    % Полупрозрачная маска неба
    mask_overlay = cat(3, zeros(h,w), zeros(h,w), double(bw));
    h_mask = imshow(mask_overlay);
    set(h_mask, 'AlphaData', bw*0.3);
    plot([1, w], [y_left, y_right], 'y-', 'LineWidth', 2);
    title('Наложение маски неба');
    
    % 6. Информация
    subplot(2,3,6);
    axis off;
    text(0.1, 0.9, 'ПОРОГОВАЯ ОБРАБОТКА', 'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.7, sprintf('Порог (Otsu): %.1f', threshold_val), 'FontSize', 11);
    text(0.1, 0.5, sprintf('Небо: %.1f%%', 100*sum(bw(:))/(h*w)), 'FontSize', 11);
    text(0.1, 0.3, sprintf('Море: %.1f%%', 100*sum(~bw(:))/(h*w)), 'FontSize', 11);
    text(0.1, 0.1, sprintf('Горизонт: y = %.2f*x + %.2f', p(1), p(2)), 'FontSize', 11);
    
    %% ВЫВОД В КОНСОЛЬ
    fprintf('\n========== ПОРОГОВАЯ ОБРАБОТКА ==========\n');
    fprintf('Порог (Otsu): %.2f (из 255)\n', threshold_val);
    fprintf('Небо: %.1f%% изображения\n', 100*sum(bw(:))/(h*w));
    fprintf('Море: %.1f%% изображения\n', 100*sum(~bw(:))/(h*w));
    fprintf('ЛИНИЯ ГОРИЗОНТА:\n');
    fprintf('  Уравнение: y = %.2f * x + %.2f\n', p(1), p(2));
    fprintf('  Слева: (1, %.1f)\n', y_left);
    fprintf('  Справа: (%d, %.1f)\n', w, y_right);
    fprintf('==========================================\n\n');
end

function horizon_kmeans_rgb(input_image_path)
    %% Загрузка и подготовка изображения
    img = imread(input_image_path);
    [h, w, ~] = size(img);
    
    %% 1. ПОДГОТОВКА ПРИЗНАКОВ [Y, X, R, G, B]
    [X, Y] = meshgrid(1:w, 1:h);
    R = double(img(:, :, 1));
    G = double(img(:, :, 2));
    B = double(img(:, :, 3));
    
    features = [Y(:), X(:), R(:), G(:), B(:)];
    features_norm = normalize(features, 'range', [0 1]);
    
    %% 2. K-MEANS (2 кластера)
    rng(42);
    [cluster_idx, cluster_centers] = kmeans(features_norm, 2, ...
                                             'Replicates', 3, ...
                                             'Display', 'off');
    
    %% 3. ОПРЕДЕЛЯЕМ, ГДЕ НЕБО (по яркости + вертикали)
    min_vals = min(features, [], 1);
    max_vals = max(features, [], 1);
    centers = cluster_centers .* (max_vals - min_vals) + min_vals;
    
    % Яркость кластеров
    bright1 = sum(centers(1, 3:5));
    bright2 = sum(centers(2, 3:5));
    
    % Небо - brighter AND higher (меньше Y)
    if bright1 > bright2
        sky_idx = 1;
    else
        sky_idx = 2;
    end
    
    % Проверяем по вертикали
    if centers(sky_idx, 1) > centers(3-sky_idx, 1)
        sky_idx = 3 - sky_idx;  % инвертируем
    end
    
    %% 4. МАСКА ВСЕГО ИЗОБРАЖЕНИЯ (1 - небо, 0 - море)
    mask = reshape(cluster_idx, h, w) == sky_idx;
    
    %% 5. МОРФОЛОГИЯ (одна операция для всей маски)
    se = strel('disk', 5);
    mask = imopen(mask, se);      % убираем белый шум в море
    mask = imclose(mask, se);     % заполняем чёрные дыры в небе
    mask = bwareaopen(mask, 100); % убираем мелкие объекты
    
    %% 6. BOUNDING BOX ДЛЯ НЕБА (из маски)
    stats = regionprops(mask, 'BoundingBox', 'Centroid', 'Area');
    [~, idx] = max([stats.Area]);
    sky_bbox = stats(idx).BoundingBox;
    sky_center = stats(idx).Centroid;
    
    %% 7. BOUNDING BOX ДЛЯ МОРЯ (инверсия маски)
    sea_stats = regionprops(~mask, 'BoundingBox', 'Centroid', 'Area');
    [~, idx] = max([sea_stats.Area]);
    sea_bbox = sea_stats(idx).BoundingBox;
    sea_center = sea_stats(idx).Centroid;
    
    %% 8. ЛИНИЯ ЦЕНТРОВ И ПЕРПЕНДИКУЛЯР
    % Вектор от моря к небу
    vec = [sky_center(1) - sea_center(1), sky_center(2) - sea_center(2)];
    if norm(vec) > 0
        vec = vec / norm(vec);
    end
    perp = [-vec(2), vec(1)];
    
    % Середина
    mid = (sky_center + sea_center) / 2;
    
    % Перпендикуляр до границ
    t = [];
    if abs(perp(1)) > 1e-6
        t = [t, (1 - mid(1))/perp(1), (w - mid(1))/perp(1)];
    end
    if abs(perp(2)) > 1e-6
        t = [t, (1 - mid(2))/perp(2), (h - mid(2))/perp(2)];
    end
    t = t(t >= min(t) & t <= max(t));
    t_min = min(t);
    t_max = max(t);
    
    horizon = [mid(1) + [t_min t_max]*perp(1); mid(2) + [t_min t_max]*perp(2)]';
    
    %% 9. ВИЗУАЛИЗАЦИЯ
    % Основная фигура
    figure('Name', 'K-means RGB', 'Position', [100 100 1200 800]);
    
    subplot(2,3,1); imshow(img); title('Исходное');
    subplot(2,3,2); imshow(label2rgb(reshape(cluster_idx, h, w))); title('Кластеризация');
    subplot(2,3,3); imshow(mask); title('Маска после морфологии');
    
    subplot(2,3,4); imshow(img); hold on;
    rectangle('Position', sky_bbox, 'EdgeColor', 'b', 'LineWidth', 2, 'LineStyle', '--');
    rectangle('Position', sea_bbox, 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '--');
    plot(sky_center(1), sky_center(2), 'b*', 'MarkerSize', 15);
    plot(sea_center(1), sea_center(2), 'r*', 'MarkerSize', 15);
    plot([sky_center(1) sea_center(1)], [sky_center(2) sea_center(2)], 'g-', 'LineWidth', 2);
    plot(mid(1), mid(2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'y');
    plot(horizon(:,1), horizon(:,2), 'y-', 'LineWidth', 1);
    title('Геометрия');
    
    subplot(2,3,5); imshow(img); hold on;
    plot(horizon(:,1), horizon(:,2), 'y-', 'LineWidth', 1);
    title('Итоговый горизонт');
    
    subplot(2,3,6); axis off;
    text(0.1, 0.9, 'ПАРАМЕТРЫ', 'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.7, sprintf('Небо: (%.0f, %.0f)', sky_center));
    text(0.1, 0.5, sprintf('Море: (%.0f, %.0f)', sea_center));
    text(0.1, 0.3, sprintf('Горизонт: (%.0f,%.0f)-(%.0f,%.0f)', horizon(1,:), horizon(2,:)));
    
    % Отдельная фигура с геометрией
    figure('Name', 'Геометрия', 'Position', [200 200 800 600]);
    imshow(img); hold on;
    rectangle('Position', sky_bbox, 'EdgeColor', 'b', 'LineWidth', 2, 'LineStyle', '--');
    rectangle('Position', sea_bbox, 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '--');
    plot(sky_center(1), sky_center(2), 'b*', 'MarkerSize', 20);
    plot(sea_center(1), sea_center(2), 'r*', 'MarkerSize', 20);
    plot([sky_center(1) sea_center(1)], [sky_center(2) sea_center(2)], 'g-', 'LineWidth', 2);
    plot(mid(1), mid(2), 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'y');
    plot(horizon(:,1), horizon(:,2), 'y-', 'LineWidth', 2.5);
    title('Центры, линия центров и перпендикуляр (горизонт)');
    
    fprintf('\n========== РЕЗУЛЬТАТ ==========\n');
    fprintf('Небо: центр (%.1f, %.1f)\n', sky_center);
    fprintf('Море: центр (%.1f, %.1f)\n', sea_center);
    fprintf('Горизонт: (%.1f,%.1f) - (%.1f,%.1f)\n', horizon(1,:), horizon(2,:));
    fprintf('===============================\n\n');
end

close all; clear; clc;

% horizon_kmeans_rgb('cadr_7.jpg');

% horizon_hough('cadr_7.jpg');
    % horizon_hough('cadr_12.jpg')
% horizon_kmeans_rgb('cadr_7.jpg');
    % horizon_kmeans_rgb('cadr_12.jpg');
% horizon_threshold('cadr_7.jpg');
    horizon_threshold('cadr_12.jpg');

% % % % % % % % % % % horizon_kmeans('cadr_7.jpg');
% % % % % % % % % % % % % % % % % %     horizon_kmeans('cadr_12.jpg')
