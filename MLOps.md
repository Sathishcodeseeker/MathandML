# FastAPI → Spring Boot Migration Guide
## Flight Congestion ML System Serving Layer

---

## Project Structure

```
flight-prediction-service/
├── pom.xml
├── src/
│   ├── main/
│   │   ├── java/com/flight/prediction/
│   │   │   ├── FlightPredictionApplication.java
│   │   │   ├── config/
│   │   │   │   ├── AppConfig.java
│   │   │   │   ├── DatabricksConfig.java
│   │   │   │   └── MLflowConfig.java
│   │   │   ├── controller/
│   │   │   │   └── PredictionController.java
│   │   │   ├── dto/
│   │   │   │   ├── PredictRequest.java
│   │   │   │   ├── PredictResponse.java
│   │   │   │   └── RouteResult.java
│   │   │   ├── service/
│   │   │   │   ├── FeatureService.java
│   │   │   │   ├── ModelService.java
│   │   │   │   └── RouteService.java
│   │   │   ├── repository/
│   │   │   │   └── DatabricksRepository.java
│   │   │   └── exception/
│   │   │       ├── GlobalExceptionHandler.java
│   │   │       └── FeatureNotFoundException.java
│   │   └── resources/
│   │       ├── application.yml
│   │       └── application-prod.yml
│   └── test/
└── Dockerfile
```

---

## 1. Dependencies (pom.xml)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.0</version>
    </parent>
    
    <groupId>com.flight</groupId>
    <artifactId>prediction-service</artifactId>
    <version>1.0.0</version>
    <name>Flight Prediction Service</name>
    
    <properties>
        <java.version>17</java.version>
        <mlflow.version>2.9.2</mlflow.version>
    </properties>
    
    <dependencies>
        <!-- Spring Boot Web -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        
        <!-- Spring Boot Validation -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-validation</artifactId>
        </dependency>
        
        <!-- Spring Boot Actuator (health checks) -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-actuator</artifactId>
        </dependency>
        
        <!-- Databricks SQL Connector -->
        <dependency>
            <groupId>com.databricks</groupId>
            <artifactId>databricks-jdbc</artifactId>
            <version>2.6.36</version>
        </dependency>
        
        <!-- MLflow Client -->
        <dependency>
            <groupId>org.mlflow</groupId>
            <artifactId>mlflow-client</artifactId>
            <version>${mlflow.version}</version>
        </dependency>
        
        <!-- Apache Commons Math (for statistical functions) -->
        <dependency>
            <groupId>org.apache.commons</groupId>
            <artifactId>commons-math3</artifactId>
            <version>3.6.1</version>
        </dependency>
        
        <!-- Lombok (reduce boilerplate) -->
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <optional>true</optional>
        </dependency>
        
        <!-- Micrometer (metrics) -->
        <dependency>
            <groupId>io.micrometer</groupId>
            <artifactId>micrometer-registry-prometheus</artifactId>
        </dependency>
        
        <!-- Testing -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>
```

---

## 2. Configuration Files

### application.yml

```yaml
spring:
  application:
    name: flight-prediction-service
  
server:
  port: 8080
  compression:
    enabled: true
  
# Databricks Configuration
databricks:
  host: ${DATABRICKS_HOST}
  token: ${DATABRICKS_TOKEN}
  http-path: ${DATABRICKS_HTTP_PATH}
  connection-pool:
    max-size: 20
    min-idle: 5
    max-lifetime: 1800000  # 30 minutes

# MLflow Configuration
mlflow:
  tracking-uri: ${MLFLOW_TRACKING_URI}
  model-name: route_congestion_ranker
  model-stage: Production
  refresh-interval-seconds: 300  # 5 minutes

# Logging
logging:
  level:
    com.flight.prediction: INFO
    org.springframework.web: WARN
  pattern:
    console: "%d{yyyy-MM-dd HH:mm:ss} - %msg%n"

# Actuator endpoints
management:
  endpoints:
    web:
      exposure:
        include: health,metrics,prometheus
  metrics:
    export:
      prometheus:
        enabled: true
```

---

## 3. Configuration Classes

### AppConfig.java

```java
package com.flight.prediction.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import lombok.Data;

@Data
@Configuration
@ConfigurationProperties(prefix = "databricks")
public class DatabricksConfig {
    private String host;
    private String token;
    private String httpPath;
    private ConnectionPool connectionPool = new ConnectionPool();
    
    @Data
    public static class ConnectionPool {
        private int maxSize = 20;
        private int minIdle = 5;
        private long maxLifetime = 1800000;
    }
    
    public String getJdbcUrl() {
        return String.format(
            "jdbc:databricks://%s:443;httpPath=%s;AuthMech=3;UID=token;PWD=%s",
            host, httpPath, token
        );
    }
}
```

### MLflowConfig.java

```java
package com.flight.prediction.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import lombok.Data;

@Data
@Configuration
@ConfigurationProperties(prefix = "mlflow")
public class MLflowConfig {
    private String trackingUri;
    private String modelName;
    private String modelStage;
    private int refreshIntervalSeconds = 300;
}
```

---

## 4. DTOs (Request/Response Objects)

### PredictRequest.java

```java
package com.flight.prediction.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import lombok.Data;
import java.time.LocalDateTime;

@Data
public class PredictRequest {
    
    @NotBlank(message = "Flight number is required")
    private String flightNumber;
    
    @NotBlank(message = "Origin is required")
    @Pattern(regexp = "^[A-Z]{3}$", message = "Origin must be a valid 3-letter IATA code")
    private String origin;
    
    @NotBlank(message = "Destination is required")
    @Pattern(regexp = "^[A-Z]{3}$", message = "Destination must be a valid 3-letter IATA code")
    private String destination;
    
    @NotNull(message = "Departure time is required")
    private LocalDateTime departureTime;
    
    @NotBlank(message = "Aircraft type is required")
    private String aircraftType;
}
```

### RouteResult.java

```java
package com.flight.prediction.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class RouteResult {
    private String route;
    private String via;
    private double congestionScore;
    private String congestionLabel;
}
```

### PredictResponse.java

```java
package com.flight.prediction.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import java.util.List;

@Data
@AllArgsConstructor
public class PredictResponse {
    private String origin;
    private String destination;
    private List<RouteResult> topRoutes;
}
```

---

## 5. Feature Service

### FeatureService.java

```java
package com.flight.prediction.service;

import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.time.DayOfWeek;
import java.util.*;

@Service
public class FeatureService {
    
    // India holidays (simplified - use proper holiday library in production)
    private static final Set<String> HOLIDAYS = Set.of(
        "2025-01-26", "2025-08-15", "2025-10-02", "2025-12-25"
    );
    
    public String getTimeBucket(int hour) {
        if (hour >= 5 && hour <= 9) return "morning_peak";
        if (hour >= 10 && hour <= 14) return "midday";
        if (hour >= 15 && hour <= 19) return "evening_peak";
        return "off_peak";
    }
    
    public Map<String, Object> computeTimeFeatures(LocalDateTime timestamp) {
        Map<String, Object> features = new HashMap<>();
        
        int hour = timestamp.getHour();
        int dayOfWeek = timestamp.getDayOfWeek().getValue(); // 1=Monday, 7=Sunday
        int month = timestamp.getMonthValue();
        
        features.put("hour_of_day", hour);
        features.put("day_of_week", dayOfWeek - 1); // 0-6 for Monday-Sunday
        features.put("month", month);
        features.put("is_weekend", (dayOfWeek >= 6) ? 1 : 0);
        features.put("is_holiday", isHoliday(timestamp) ? 1 : 0);
        features.put("season", getSeason(month));
        
        // Cyclical encoding
        features.put("hour_sin", Math.sin(2 * Math.PI * hour / 24.0));
        features.put("hour_cos", Math.cos(2 * Math.PI * hour / 24.0));
        features.put("day_sin", Math.sin(2 * Math.PI * (dayOfWeek - 1) / 7.0));
        features.put("day_cos", Math.cos(2 * Math.PI * (dayOfWeek - 1) / 7.0));
        
        return features;
    }
    
    private int getSeason(int month) {
        if (month == 12 || month <= 2) return 0; // winter
        if (month >= 3 && month <= 5) return 1;  // spring
        if (month >= 6 && month <= 8) return 2;  // monsoon
        return 3; // autumn
    }
    
    private boolean isHoliday(LocalDateTime date) {
        String dateStr = date.toLocalDate().toString();
        return HOLIDAYS.contains(dateStr);
    }
    
    public String getCongestionLabel(double score) {
        if (score < 0.33) return "LOW";
        if (score < 0.66) return "MEDIUM";
        return "HIGH";
    }
}
```

---

## 6. Databricks Repository

### DatabricksRepository.java

```java
package com.flight.prediction.repository;

import com.flight.prediction.config.DatabricksConfig;
import com.flight.prediction.exception.FeatureNotFoundException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Repository;

import java.sql.*;
import java.util.*;

@Slf4j
@Repository
public class DatabricksRepository {
    
    private final DatabricksConfig config;
    
    public DatabricksRepository(DatabricksConfig config) {
        this.config = config;
    }
    
    private Connection getConnection() throws SQLException {
        return DriverManager.getConnection(config.getJdbcUrl());
    }
    
    public Map<String, Object> fetchAirportFeatures(
            String airport, String timeBucket) {
        
        String query = """
            SELECT runway_utilization, avg_departure_delay,
                   gate_occupancy_rate, weather_severity,
                   visibility_score, wind_severity,
                   active_departures, cancellation_rate
            FROM gold.airport_features
            WHERE airport_code = ? AND time_bucket = ?
            LIMIT 1
        """;
        
        try (Connection conn = getConnection();
             PreparedStatement stmt = conn.prepareStatement(query)) {
            
            stmt.setString(1, airport);
            stmt.setString(2, timeBucket);
            
            ResultSet rs = stmt.executeQuery();
            
            if (!rs.next()) {
                throw new FeatureNotFoundException(
                    String.format("No features for airport %s / %s", 
                                  airport, timeBucket));
            }
            
            Map<String, Object> features = new HashMap<>();
            features.put("runway_utilization", rs.getDouble(1));
            features.put("avg_departure_delay", rs.getDouble(2));
            features.put("gate_occupancy_rate", rs.getDouble(3));
            features.put("weather_severity", rs.getDouble(4));
            features.put("visibility_score", rs.getDouble(5));
            features.put("wind_severity", rs.getDouble(6));
            features.put("active_departures", rs.getInt(7));
            features.put("cancellation_rate", rs.getDouble(8));
            
            return features;
            
        } catch (SQLException e) {
            log.error("Database error fetching airport features", e);
            throw new RuntimeException("Database error", e);
        }
    }
    
    public Map<String, Object> fetchRouteFeatures(
            String origin, String destination, String via, String timeBucket) {
        
        String query = """
            SELECT route_distance_km, airspace_sector_load,
                   historical_route_delay, route_on_time_rate,
                   route_flight_frequency
            FROM gold.route_features
            WHERE origin = ? AND destination = ? 
              AND via_airport = ? AND time_bucket = ?
            LIMIT 1
        """;
        
        try (Connection conn = getConnection();
             PreparedStatement stmt = conn.prepareStatement(query)) {
            
            stmt.setString(1, origin);
            stmt.setString(2, destination);
            stmt.setString(3, via);
            stmt.setString(4, timeBucket);
            
            ResultSet rs = stmt.executeQuery();
            
            if (!rs.next()) {
                throw new FeatureNotFoundException(
                    String.format("No route features for %s→%s→%s", 
                                  origin, via, destination));
            }
            
            Map<String, Object> features = new HashMap<>();
            features.put("route_distance_km", rs.getDouble(1));
            features.put("airspace_sector_load", rs.getDouble(2));
            features.put("historical_route_delay", rs.getDouble(3));
            features.put("route_on_time_rate", rs.getDouble(4));
            features.put("route_flight_frequency", rs.getInt(5));
            
            return features;
            
        } catch (SQLException e) {
            log.error("Database error fetching route features", e);
            throw new RuntimeException("Database error", e);
        }
    }
    
    public List<String> fetchPossibleVias(String origin, String destination) {
        String query = """
            SELECT DISTINCT via_airport 
            FROM gold.route_features
            WHERE origin = ? AND destination = ?
            ORDER BY route_flight_frequency DESC
        """;
        
        List<String> vias = new ArrayList<>();
        
        try (Connection conn = getConnection();
             PreparedStatement stmt = conn.prepareStatement(query)) {
            
            stmt.setString(1, origin);
            stmt.setString(2, destination);
            
            ResultSet rs = stmt.executeQuery();
            while (rs.next()) {
                vias.add(rs.getString(1));
            }
            
            return vias;
            
        } catch (SQLException e) {
            log.error("Database error fetching vias", e);
            throw new RuntimeException("Database error", e);
        }
    }
}
```

---

## 7. Model Service (Calling Databricks Model Serving Endpoint)

### Updated MLflowConfig.java

```java
package com.flight.prediction.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import lombok.Data;

@Data
@Configuration
@ConfigurationProperties(prefix = "mlflow")
public class MLflowConfig {
    private String trackingUri;
    private String modelName;
    private String modelStage;
    private String servingEndpoint;  // NEW: Databricks Model Serving URL
    private String token;             // NEW: Databricks PAT token
}
```

### Updated application.yml

```yaml
mlflow:
  tracking-uri: ${MLFLOW_TRACKING_URI}
  model-name: route_congestion_ranker
  model-stage: Production
  serving-endpoint: ${DATABRICKS_SERVING_ENDPOINT}  # e.g., https://adb-xxx.azuredatabricks.net/serving-endpoints/route_ranker/invocations
  token: ${DATABRICKS_TOKEN}
```

### ModelService.java (REST Client Version)

```java
package com.flight.prediction.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.flight.prediction.config.MLflowConfig;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;

import java.util.*;

@Slf4j
@Service
public class ModelService {
    
    private final MLflowConfig config;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    
    public ModelService(MLflowConfig config, RestTemplate restTemplate) {
        this.config = config;
        this.restTemplate = restTemplate;
        this.objectMapper = new ObjectMapper();
    }
    
    /**
     * Call Databricks Model Serving endpoint for batch predictions
     */
    public List<Double> predict(List<Map<String, Object>> featureRows) {
        try {
            // Build request payload
            Map<String, Object> payload = new HashMap<>();
            payload.put("dataframe_records", featureRows);
            
            // Set headers
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + config.getToken());
            
            HttpEntity<Map<String, Object>> request = 
                new HttpEntity<>(payload, headers);
            
            // Call endpoint
            log.debug("Calling model endpoint: {}", config.getServingEndpoint());
            
            ResponseEntity<Map> response = restTemplate.exchange(
                config.getServingEndpoint(),
                HttpMethod.POST,
                request,
                Map.class
            );
            
            // Parse predictions
            if (response.getStatusCode() == HttpStatus.OK) {
                Map<String, Object> responseBody = response.getBody();
                List<Double> predictions = (List<Double>) responseBody.get("predictions");
                
                log.info("Received {} predictions from model", predictions.size());
                return predictions;
            } else {
                throw new RuntimeException(
                    "Model endpoint returned status: " + response.getStatusCode()
                );
            }
            
        } catch (Exception e) {
            log.error("Error calling model serving endpoint", e);
            throw new RuntimeException("Prediction failed", e);
        }
    }
    
    /**
     * Health check for model endpoint
     */
    public boolean isModelEndpointHealthy() {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.set("Authorization", "Bearer " + config.getToken());
            
            HttpEntity<String> request = new HttpEntity<>(headers);
            
            // Most Databricks endpoints expose /health or you can do a minimal prediction
            ResponseEntity<String> response = restTemplate.exchange(
                config.getServingEndpoint(),
                HttpMethod.GET,
                request,
                String.class
            );
            
            return response.getStatusCode() == HttpStatus.OK;
            
        } catch (Exception e) {
            log.error("Model endpoint health check failed", e);
            return false;
        }
    }
    
    public String getCurrentModelEndpoint() {
        return config.getServingEndpoint();
    }
}
```

### RestTemplate Configuration

```java
package com.flight.prediction.config;

import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;

@Configuration
public class RestClientConfig {
    
    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder builder) {
        return builder
            .setConnectTimeout(Duration.ofSeconds(10))
            .setReadTimeout(Duration.ofSeconds(30))
            .build();
    }
}
```

---

## Databricks Model Serving Setup

### 1. Register Model in MLflow (from your training notebook)

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    # Train your pipeline
    pipeline.fit(X_train, y_train)
    
    # Log model
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="route_ranker_pipeline",
        registered_model_name="route_congestion_ranker",
        input_example=X_train.head(1)  # IMPORTANT: helps define schema
    )
```

### 2. Deploy Model to Serving Endpoint

**Via Databricks UI:**
1. Go to **Machine Learning** → **Serving**
2. Click **Create Serving Endpoint**
3. Name: `route_ranker`
4. Model: `route_congestion_ranker`
5. Version: Select your production version
6. Cluster size: Small (for dev) or Medium/Large (for prod)
7. Click **Create**

**Via Databricks REST API:**

```bash
curl -X POST \
  https://your-workspace.databricks.com/api/2.0/serving-endpoints \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "route_ranker",
    "config": {
      "served_models": [{
        "model_name": "route_congestion_ranker",
        "model_version": "1",
        "workload_size": "Small",
        "scale_to_zero_enabled": false
      }]
    }
  }'
```

### 3. Get Endpoint URL

After deployment, your endpoint URL will be:
```
https://<workspace-url>/serving-endpoints/route_ranker/invocations
```

### 4. Test the Endpoint

**Request:**
```bash
curl -X POST \
  https://your-workspace.databricks.com/serving-endpoints/route_ranker/invocations \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {
        "origin": "BLR",
        "destination": "DEL",
        "via_airport": "BOM",
        "aircraft_type": "B737",
        "airline_code": "AI",
        "time_bucket": "morning_peak",
        "runway_utilization": 0.87,
        "avg_departure_delay": 12.4,
        "gate_occupancy_rate": 0.76,
        "weather_severity": 2,
        "visibility_score": 0.85,
        "wind_severity": 1,
        "active_departures": 45,
        "cancellation_rate": 0.03,
        "route_distance_km": 1742,
        "airspace_sector_load": 0.68,
        "historical_route_delay": 8.2,
        "route_on_time_rate": 0.72,
        "route_flight_frequency": 12,
        "runway_util_dest": 0.61,
        "avg_delay_dest": 4.1,
        "weather_dest": 2,
        "hour_of_day": 8,
        "day_of_week": 1,
        "month": 2,
        "is_weekend": 0,
        "is_holiday": 0,
        "season": 0,
        "hour_sin": 0.866,
        "hour_cos": 0.5,
        "day_sin": 0.433,
        "day_cos": 0.901
      },
      {
        "origin": "BLR",
        "destination": "DEL",
        "via_airport": "AMD",
        "aircraft_type": "B737",
        "airline_code": "AI",
        "time_bucket": "morning_peak",
        "runway_utilization": 0.87,
        "avg_departure_delay": 12.4,
        "gate_occupancy_rate": 0.76,
        "weather_severity": 2,
        "visibility_score": 0.85,
        "wind_severity": 1,
        "active_departures": 45,
        "cancellation_rate": 0.03,
        "route_distance_km": 1893,
        "airspace_sector_load": 0.82,
        "historical_route_delay": 15.7,
        "route_on_time_rate": 0.58,
        "route_flight_frequency": 8,
        "runway_util_dest": 0.61,
        "avg_delay_dest": 4.1,
        "weather_dest": 2,
        "hour_of_day": 8,
        "day_of_week": 1,
        "month": 2,
        "is_weekend": 0,
        "is_holiday": 0,
        "season": 0,
        "hour_sin": 0.866,
        "hour_cos": 0.5,
        "day_sin": 0.433,
        "day_cos": 0.901
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [0.28, 0.72]
}
```

### 5. Monitor Endpoint Performance

Databricks automatically tracks:
- **Request latency** (p50, p95, p99)
- **Request volume** (requests/second)
- **Error rate**
- **Model version** being served

Access metrics at:
```
https://<workspace>/ml/endpoints/route_ranker
```

### 6. Update Model (Zero-Downtime)

When you train a new model version:

```python
# Train new version
with mlflow.start_run():
    new_pipeline.fit(X_train, y_train)
    mlflow.sklearn.log_model(new_pipeline, "route_ranker_pipeline",
                             registered_model_name="route_congestion_ranker")
    # This becomes version 2
```

Update endpoint to serve new version:

```bash
curl -X PUT \
  https://your-workspace.databricks.com/api/2.0/serving-endpoints/route_ranker/config \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "served_models": [{
      "model_name": "route_congestion_ranker",
      "model_version": "2",
      "workload_size": "Small",
      "scale_to_zero_enabled": false
    }]
  }'
```

Databricks will:
1. Spin up new instances with v2
2. Gradually shift traffic from v1 → v2
3. Shut down v1 instances once v2 is stable
4. **Zero downtime** for your Spring Boot service!

---

## Updated pom.xml (Remove XGBoost, Add Jackson)

```xml
<dependencies>
    <!-- Spring Boot Web (includes RestTemplate) -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    
    <!-- Spring Boot Validation -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-validation</artifactId>
    </dependency>
    
    <!-- Spring Boot Actuator -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-actuator</artifactId>
    </dependency>
    
    <!-- Databricks SQL Connector -->
    <dependency>
        <groupId>com.databricks</groupId>
        <artifactId>databricks-jdbc</artifactId>
        <version>2.6.36</version>
    </dependency>
    
    <!-- Jackson for JSON (included in spring-boot-starter-web) -->
    
    <!-- Lombok -->
    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
        <optional>true</optional>
    </dependency>
    
    <!-- Apache Commons Math (if needed) -->
    <dependency>
        <groupId>org.apache.commons</groupId>
        <artifactId>commons-math3</artifactId>
        <version>3.6.1</version>
    </dependency>
</dependencies>
```

**Note:** You no longer need XGBoost4J, MLflow client, or ONNX dependencies!

---

## 8. Controller

### PredictionController.java

```java
package com.flight.prediction.controller;

import com.flight.prediction.dto.*;
import com.flight.prediction.service.*;
import com.flight.prediction.repository.DatabricksRepository;
import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@RestController
@RequestMapping("/api/v1")
public class PredictionController {
    
    private final FeatureService featureService;
    private final ModelService modelService;
    private final DatabricksRepository databricksRepo;
    
    public PredictionController(
            FeatureService featureService,
            ModelService modelService,
            DatabricksRepository databricksRepo) {
        this.featureService = featureService;
        this.modelService = modelService;
        this.databricksRepo = databricksRepo;
    }
    
    @PostMapping("/predict/top-routes")
    public ResponseEntity<PredictResponse> predictTopRoutes(
            @Valid @RequestBody PredictRequest request) {
        
        log.info("Prediction request: {} → {}", 
                 request.getOrigin(), request.getDestination());
        
        String timeBucket = featureService.getTimeBucket(
            request.getDepartureTime().getHour()
        );
        
        String origin = request.getOrigin();
        String destination = request.getDestination();
        String airline = request.getFlightNumber().substring(0, 2);
        
        // Fetch shared features
        Map<String, Object> originFeatures = 
            databricksRepo.fetchAirportFeatures(origin, timeBucket);
        Map<String, Object> destFeatures = 
            databricksRepo.fetchAirportFeatures(destination, timeBucket);
        Map<String, Object> timeFeatures = 
            featureService.computeTimeFeatures(request.getDepartureTime());
        
        // Get possible via airports
        List<String> possibleVias = 
            databricksRepo.fetchPossibleVias(origin, destination);
        
        if (possibleVias.isEmpty()) {
            return ResponseEntity.notFound().build();
        }
        
        // Build feature rows for each route
        List<Map<String, Object>> featureRows = new ArrayList<>();
        List<String> validVias = new ArrayList<>();
        
        for (String via : possibleVias) {
            try {
                Map<String, Object> routeFeatures = 
                    databricksRepo.fetchRouteFeatures(
                        origin, destination, via, timeBucket);
                
                Map<String, Object> fullFeatures = new HashMap<>();
                fullFeatures.put("origin", origin);
                fullFeatures.put("destination", destination);
                fullFeatures.put("via_airport", via);
                fullFeatures.put("aircraft_type", request.getAircraftType());
                fullFeatures.put("airline_code", airline);
                fullFeatures.put("time_bucket", timeBucket);
                
                // Add all feature groups
                fullFeatures.putAll(originFeatures);
                fullFeatures.putAll(routeFeatures);
                fullFeatures.put("runway_util_dest", 
                               destFeatures.get("runway_utilization"));
                fullFeatures.put("avg_delay_dest", 
                               destFeatures.get("avg_departure_delay"));
                fullFeatures.put("weather_dest", 
                               destFeatures.get("weather_severity"));
                fullFeatures.putAll(timeFeatures);
                
                featureRows.add(fullFeatures);
                validVias.add(via);
                
            } catch (Exception e) {
                log.warn("Skipping via {} due to missing features", via);
            }
        }
        
        // Predict all routes
        List<Double> scores = modelService.predict(featureRows);
        
        // Build results
        List<RouteResult> results = new ArrayList<>();
        for (int i = 0; i < validVias.size(); i++) {
            String via = validVias.get(i);
            double score = scores.get(i);
            
            results.add(new RouteResult(
                String.format("%s → %s → %s", origin, via, destination),
                via,
                Math.round(score * 1000.0) / 1000.0,
                featureService.getCongestionLabel(score)
            ));
        }
        
        // Sort by score (ascending - lower is better)
        results.sort(Comparator.comparingDouble(RouteResult::getCongestionScore));
        
        // Return top 3
        List<RouteResult> topRoutes = results.stream()
            .limit(3)
            .collect(Collectors.toList());
        
        return ResponseEntity.ok(
            new PredictResponse(origin, destination, topRoutes)
        );
    }
    
    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> healthCheck() {
        Map<String, String> health = new HashMap<>();
        health.put("status", "UP");
        health.put("model_version", modelService.getCurrentModelVersion());
        return ResponseEntity.ok(health);
    }
}
```

---

## 9. Exception Handling

### GlobalExceptionHandler.java

```java
package com.flight.prediction.exception;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.HashMap;
import java.util.Map;

@RestControllerAdvice
public class GlobalExceptionHandler {
    
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<Map<String, String>> handleValidationErrors(
            MethodArgumentNotValidException ex) {
        
        Map<String, String> errors = new HashMap<>();
        ex.getBindingResult().getAllErrors().forEach((error) -> {
            String fieldName = ((FieldError) error).getField();
            String errorMessage = error.getDefaultMessage();
            errors.put(fieldName, errorMessage);
        });
        
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errors);
    }
    
    @ExceptionHandler(FeatureNotFoundException.class)
    public ResponseEntity<Map<String, String>> handleFeatureNotFound(
            FeatureNotFoundException ex) {
        
        Map<String, String> error = new HashMap<>();
        error.put("error", ex.getMessage());
        
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(error);
    }
    
    @ExceptionHandler(Exception.class)
    public ResponseEntity<Map<String, String>> handleGenericError(
            Exception ex) {
        
        Map<String, String> error = new HashMap<>();
        error.put("error", "Internal server error");
        error.put("message", ex.getMessage());
        
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(error);
    }
}
```

### FeatureNotFoundException.java

```java
package com.flight.prediction.exception;

public class FeatureNotFoundException extends RuntimeException {
    public FeatureNotFoundException(String message) {
        super(message);
    }
}
```

---

## 10. Main Application

### FlightPredictionApplication.java

```java
package com.flight.prediction;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
public class FlightPredictionApplication {
    
    public static void main(String[] args) {
        SpringApplication.run(FlightPredictionApplication.class, args);
    }
}
```

---

## 11. Build & Run

### Build with Maven

```bash
mvn clean package
```

### Run locally

```bash
export DATABRICKS_HOST=your-workspace.databricks.com
export DATABRICKS_TOKEN=your-token
export DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/abc123
export MLFLOW_TRACKING_URI=databricks

java -jar target/prediction-service-1.0.0.jar
```

### Dockerfile

```dockerfile
FROM eclipse-temurin:17-jre-alpine

WORKDIR /app

COPY target/prediction-service-1.0.0.jar app.jar

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  flight-prediction:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABRICKS_HOST=${DATABRICKS_HOST}
      - DATABRICKS_TOKEN=${DATABRICKS_TOKEN}
      - DATABRICKS_HTTP_PATH=${DATABRICKS_HTTP_PATH}
      - MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
    healthcheck:
      test: ["CMD", "wget", "--spider", "http://localhost:8080/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 12. Testing

### Example Test

```java
package com.flight.prediction.controller;

import com.flight.prediction.dto.PredictRequest;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
class PredictionControllerTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    @Test
    void testPredictTopRoutes() throws Exception {
        PredictRequest request = new PredictRequest();
        request.setFlightNumber("AI101");
        request.setOrigin("BLR");
        request.setDestination("DEL");
        request.setDepartureTime(LocalDateTime.now().plusHours(2));
        request.setAircraftType("B737");
        
        mockMvc.perform(post("/api/v1/predict/top-routes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.origin").value("BLR"))
                .andExpect(jsonPath("$.destination").value("DEL"))
                .andExpect(jsonPath("$.topRoutes").isArray());
    }
}
```

---

## Key Differences: FastAPI vs Spring Boot

| Aspect | FastAPI | Spring Boot |
|--------|---------|-------------|
| **Language** | Python | Java |
| **Type Safety** | Pydantic models | Java classes + Bean Validation |
| **Dependency Injection** | Manual | Automatic via @Autowired |
| **Config Management** | .env + Pydantic | application.yml + @ConfigurationProperties |
| **Background Tasks** | Threading module | @Scheduled annotation |
| **Model Serving** | Load scikit-learn pipeline locally | **Call Databricks REST endpoint** ✅ |
| **Database** | Direct JDBC in Python | Spring JDBC with connection pooling |
| **Validation** | Pydantic validators | Jakarta Validation (@Valid) |
| **Health Checks** | Manual endpoint | Spring Actuator |
| **Metrics** | Manual | Micrometer + Prometheus |
| **Build Tool** | pip | Maven / Gradle |
| **Startup Time** | ~1-2 seconds | ~10-15 seconds |
| **Memory** | Lower | Higher |
| **Ecosystem** | Python ML libraries | Enterprise Java ecosystem |

## Why Databricks Model Serving is Better

✅ **No local model loading** - Model lives in Databricks, not in your JVM
✅ **Auto-scaling** - Databricks handles traffic spikes
✅ **Zero-downtime updates** - Deploy new models without restarting your service
✅ **Simpler code** - Just HTTP calls via RestTemplate
✅ **Same for Python or Java** - Language-agnostic inference
✅ **Built-in monitoring** - Databricks tracks latency, errors, throughput
✅ **A/B testing** - Route traffic to multiple model versions

---

## Production Considerations

### 1. Connection Pooling

```java
@Configuration
public class DataSourceConfig {
    
    @Bean
    public HikariDataSource dataSource(DatabricksConfig config) {
        HikariConfig hikariConfig = new HikariConfig();
        hikariConfig.setJdbcUrl(config.getJdbcUrl());
        hikariConfig.setMaximumPoolSize(config.getConnectionPool().getMaxSize());
        hikariConfig.setMinimumIdle(config.getConnectionPool().getMinIdle());
        hikariConfig.setMaxLifetime(config.getConnectionPool().getMaxLifetime());
        return new HikariDataSource(hikariConfig);
    }
}
```

### 2. Caching

```java
@EnableCaching
@Configuration
public class CacheConfig {
    
    @Bean
    public CacheManager cacheManager() {
        return new ConcurrentMapCacheManager("airportFeatures", "routeFeatures");
    }
}

// In service:
@Cacheable(value = "airportFeatures", key = "#airport + '-' + #timeBucket")
public Map<String, Object> fetchAirportFeatures(String airport, String timeBucket) {
    // ...
}
```

### 3. Async Processing

```java
@EnableAsync
@Configuration
public class AsyncConfig {
    
    @Bean
    public Executor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(10);
        executor.setMaxPoolSize(20);
        executor.setQueueCapacity(100);
        executor.initialize();
        return executor;
    }
}
```

### 4. Metrics

```java
@Component
public class PredictionMetrics {
    
    private final MeterRegistry registry;
    private final Counter predictionCounter;
    private final Timer predictionTimer;
    
    public PredictionMetrics(MeterRegistry registry) {
        this.registry = registry;
        this.predictionCounter = Counter.builder("predictions.total")
            .tag("service", "flight-prediction")
            .register(registry);
        this.predictionTimer = Timer.builder("predictions.duration")
            .register(registry);
    }
    
    public void recordPrediction() {
        predictionCounter.increment();
    }
    
    public Timer.Sample startTimer() {
        return Timer.start(registry);
    }
}
```

---

## Summary

### Spring Boot with Databricks Model Serving provides:

✅ **Zero ML complexity** - No XGBoost4J, PMML, or ONNX needed
✅ **Simple HTTP calls** - Just RestTemplate + JSON
✅ **Enterprise features** - Strong typing, DI, monitoring
✅ **Scalable inference** - Let Databricks handle model serving
✅ **Easy updates** - Deploy new models without redeploying your service
✅ **Language flexibility** - Same endpoint works for Java, Python, Node.js, etc.

### Architecture Flow:

```
User Request
    ↓
Spring Boot Controller (validate input)
    ↓
DatabricksRepository (fetch features from Gold tables)
    ↓
FeatureService (compute time features, build feature vectors)
    ↓
ModelService (HTTP POST to Databricks endpoint)
    ↓
Databricks Model Serving (runs XGBoost pipeline)
    ↓
Parse predictions → rank routes → return top 3
```

### What Changed from FastAPI:

1. **No local model loading** - Previously loaded scikit-learn pipeline in memory
2. **HTTP-based inference** - Call Databricks REST endpoint instead
3. **Same architecture otherwise** - Feature engineering, database queries, ranking logic all identical

### Best of Both Worlds:

- **Spring Boot strengths**: Type safety, DI, enterprise monitoring
- **Databricks strengths**: Managed inference, auto-scaling, model versioning
- **Result**: Clean separation of concerns, easier ops, better scalability

**Recommendation**: This is the **optimal architecture** for production ML systems. Your Spring Boot service handles business logic, Databricks handles ML inference. Each does what it's best at.
