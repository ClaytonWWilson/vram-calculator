# Build stage
FROM oven/bun:1 AS build

COPY ./web /app
WORKDIR /app
RUN bun install
RUN bun run build

# Runtime stage
FROM oven/bun:1

WORKDIR /app
COPY --from=build /app/build ./dist
COPY --from=build /app/node_modules ./node_modules

EXPOSE 3000
CMD ["bun", "run", "dist/index.js"]
