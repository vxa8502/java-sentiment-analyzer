/**
 * API layer tests for the Sentiment Analyzer.
 *
 * <h2>Test Categories</h2>
 *
 * <table border="1">
 *   <tr>
 *     <th>Category</th>
 *     <th>Tag</th>
 *     <th>Suffix</th>
 *     <th>Speed</th>
 *     <th>Dependencies</th>
 *   </tr>
 *   <tr>
 *     <td><b>Web Layer</b></td>
 *     <td>{@code @Tag("web")}</td>
 *     <td>{@code *WebTest}</td>
 *     <td>Fast (~1s)</td>
 *     <td>Mocked services</td>
 *   </tr>
 *   <tr>
 *     <td><b>Integration</b></td>
 *     <td>{@code @Tag("integration")}</td>
 *     <td>{@code *IntegrationTest}</td>
 *     <td>Slow (~10s)</td>
 *     <td>Real model required</td>
 *   </tr>
 * </table>
 *
 * <h2>Running Tests by Category</h2>
 * <pre>
 * # Run only web tests (fast, no model needed)
 * mvn test -Dgroups=web
 *
 * # Run only integration tests (requires model)
 * mvn test -Dgroups=integration
 *
 * # Run all tests
 * mvn test
 * </pre>
 *
 * <h2>Test Naming Convention</h2>
 * <ul>
 *   <li>{@code *WebTest} - Uses {@code @WebMvcTest}, mocked dependencies</li>
 *   <li>{@code *IntegrationTest} - Uses {@code @SpringBootTest}, real dependencies</li>
 *   <li>{@code *Test} (other) - Unit tests, no Spring context</li>
 * </ul>
 *
 * @see SentimentControllerWebTest Web layer tests (fast)
 * @see SentimentControllerIntegrationTest Integration tests (requires model)
 */
package sentiment.api;
