# Multi-stage build for optimized Java container
# Stage 1: Build the application
FROM maven:3.9-eclipse-temurin-21 AS build

WORKDIR /app

# Copy pom.xml and download dependencies (cached layer)
COPY pom.xml .
RUN mvn dependency:go-offline -B

# Copy source code and build
COPY src ./src
RUN mvn clean package -DskipTests -B

# Stage 2: Create the runtime image
FROM eclipse-temurin:21-jre

WORKDIR /app

# Install curl for health checks (with dpkg fix for package conflicts)
RUN apt-get update && \
    apt-get install -y --fix-broken && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Add non-root user for security
RUN groupadd -g 1001 appuser && \
    useradd -r -u 1001 -g appuser appuser

# Copy the JAR from build stage
COPY --from=build /app/target/sentiment-analyzer-1.0.0.jar app.jar

# Copy production model (always included in image)
COPY models/production /app/models/production

# Copy dataset documentation
COPY datasets /app/datasets

# Create data directory and set permissions
RUN mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1

# Set configurable memory limits with sensible defaults
ENV MAX_HEAP=512m
ENV MIN_HEAP=256m

# Set JVM options for container environment
ENV JAVA_OPTS="-Xmx${MAX_HEAP} -Xms${MIN_HEAP} -XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0"

# Run the application
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar /app/app.jar"]
