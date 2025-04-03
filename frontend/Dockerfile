# Этап сборки
FROM node:18-alpine AS builder
WORKDIR /app

# Копируем package.json и package-lock.json для установки зависимостей
COPY package.json package-lock.json ./
RUN npm install

# Копируем исходный код и выполняем сборку проекта
COPY . .
RUN npm run build

# Этап выполнения: используем минимальный образ для запуска собранного приложения
FROM node:18-alpine AS runner
WORKDIR /app

# Устанавливаем пакет serve для статического сервера
RUN npm install -g serve

# Копируем собранные файлы из этапа сборки
COPY --from=builder /app/dist ./dist

# Открываем порт, на котором будет запущен сервер (например, 3000)
EXPOSE 3000

# Запускаем сервер, обслуживающий содержимое папки dist
CMD ["serve", "-s", "dist", "-l", "3000"]
