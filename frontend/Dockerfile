# Build stage
FROM node:18-slim AS build

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy source code and build
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx config if needed, or use default
# For simple use, we can use default and handle API URL via env in browser

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
