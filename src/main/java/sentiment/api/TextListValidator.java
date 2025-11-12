package sentiment.api;

import jakarta.validation.ConstraintValidator;
import jakarta.validation.ConstraintValidatorContext;
import sentiment.util.ValidationUtils;

import java.util.List;

import static sentiment.api.ValidationConstants.*;

/**
 * Validator implementation for @ValidTextList annotation.
 *
 * Validates each text in a list to ensure:
 * 1. No null or blank texts
 * 2. Each text is within min/max length bounds
 *
 * This prevents bypassing single-request validation by sending
 * oversized texts in batch requests.
 */
public class TextListValidator implements ConstraintValidator<ValidTextList, List<String>> {

    @Override
    public void initialize(ValidTextList constraintAnnotation) {
        // No initialization needed
    }

    @Override
    public boolean isValid(List<String> texts, ConstraintValidatorContext context) {
        if (texts == null || texts.isEmpty()) {
            return true; // Let @NotEmpty handle null/empty validation
        }

        // Validate each text in the list
        for (int i = 0; i < texts.size(); i++) {
            String text = texts.get(i);

            // Check for null or blank
            if (ValidationUtils.isNullOrEmpty(text)) {
                context.disableDefaultConstraintViolation();
                context.buildConstraintViolationWithTemplate(
                    String.format("Text at index %d is blank or null", i)
                ).addConstraintViolation();
                return false;
            }

            // Check minimum length
            if (text.length() < MIN_TEXT_LENGTH) {
                context.disableDefaultConstraintViolation();
                context.buildConstraintViolationWithTemplate(
                    String.format("Text at index %d is too short (minimum %d characters)",
                        i, MIN_TEXT_LENGTH)
                ).addConstraintViolation();
                return false;
            }

            // Check maximum length
            if (text.length() > MAX_TEXT_LENGTH) {
                context.disableDefaultConstraintViolation();
                context.buildConstraintViolationWithTemplate(
                    String.format("Text at index %d exceeds maximum length of %d characters (got %d)",
                        i, MAX_TEXT_LENGTH, text.length())
                ).addConstraintViolation();
                return false;
            }
        }

        return true;
    }
}
