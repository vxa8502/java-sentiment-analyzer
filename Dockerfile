# Multi-stage build for optimized Java container
# Stage 1: Build the application
FROM maven:3.9-eclipse-temurin-21 AS build

WORKDIR /app

# Copy pom.xml and download dependencies (cached layer)
COPY pom.xml .
RUN mvn dependency:go-offline -B

# Copy source code and build
COPY src ./src
RUN mvn clean package -DskipTests -B && \
    mv target/sentiment-analyzer-*.jar target/app.jar

# Stage 2: Create the runtime image
FROM eclipse-temurin:21-jre

WORKDIR /app

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Add non-root user for security
RUN groupadd -g 1001 appuser && \
    useradd -r -u 1001 -g appuser appuser

# Copy the JAR from build stage
COPY --from=build /app/target/app.jar app.jar

# Copy production model (requires models/production/ to exist - run scripts/promote_to_production.sh first)
COPY models/production /app/models/production

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Default memory limits (override at runtime with -e MAX_HEAP=1g)
ENV MAX_HEAP=512m
ENV MIN_HEAP=256m

# Run the application (JAVA_OPTS constructed at runtime so ENV overrides work)
ENTRYPOINT ["sh", "-c", "java -Xmx${MAX_HEAP} -Xms${MIN_HEAP} -XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0 -jar /app/app.jar"]
