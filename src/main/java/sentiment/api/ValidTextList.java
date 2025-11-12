package sentiment.api;

import jakarta.validation.Constraint;
import jakarta.validation.Payload;

import java.lang.annotation.*;

/**
 * Custom validation annotation for validating a list of text strings.
 *
 * Ensures that each text in the list:
 * - Is not null or blank
 * - Does not exceed maximum length
 * - Meets minimum length requirement
 *
 * Used for batch request validation to prevent oversized individual texts
 * from bypassing single-request validation limits.
 */
@Target({ElementType.FIELD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = TextListValidator.class)
@Documented
public @interface ValidTextList {

    String message() default "Invalid text in list";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
