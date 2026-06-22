/**
 * REST API layer for the Sentiment Analyzer service.
 *
 * <h2>Package Contents</h2>
 * <ul>
 *   <li><b>Controllers</b>: REST endpoints ({@code *Controller})</li>
 *   <li><b>Services</b>: Business logic ({@code *Service})</li>
 *   <li><b>DTOs</b>: Request/Response objects ({@code *Request}, {@code *Response})</li>
 *   <li><b>Exception Handlers</b>: Global error handling</li>
 * </ul>
 *
 * <h2>Naming Conventions</h2>
 *
 * <h3>Accessor Methods</h3>
 * <p>This package uses two accessor patterns based on the class type:
 *
 * <table border="1">
 *   <tr><th>Type</th><th>Pattern</th><th>Example</th></tr>
 *   <tr>
 *     <td>Records (DTOs)</td>
 *     <td>Method-style: {@code fieldName()}</td>
 *     <td>{@code response.sentiment()}, {@code request.text()}</td>
 *   </tr>
 *   <tr>
 *     <td>Classes (Services, Config)</td>
 *     <td>JavaBean-style: {@code getFieldName()}</td>
 *     <td>{@code service.getClassifier()}, {@code props.getMaxFeatures()}</td>
 *   </tr>
 * </table>
 *
 * <p><b>Rationale:</b> Records automatically generate method-style accessors.
 * We accept this Java convention rather than fighting it. Traditional classes
 * continue to use JavaBean-style getters for framework compatibility.
 *
 * <h3>DTO Naming</h3>
 * <ul>
 *   <li>Request DTOs: {@code *Request} (e.g., {@link SentimentRequest})</li>
 *   <li>Response DTOs: {@code *Response} (e.g., {@link SentimentResponse})</li>
 *   <li>All DTOs are immutable records</li>
 * </ul>
 *
 * <h2>Error Handling Pattern</h2>
 * <p>Domain responses (e.g., {@link SentimentResponse}) include an {@code error}
 * field for domain-specific errors. Generic API errors use {@link ErrorResponse}.
 *
 * @see SentimentController Main sentiment analysis endpoints
 * @see ErrorResponse Standard error response format
 */
package sentiment.api;
