package sentiment.data;

/**
 * Custom exception for data loading errors across all dataset loaders.
 * Provides structured error information including file path, dataset type,
 * and detailed error context for better debugging and user feedback.
 */
public class DataLoadingException extends Exception {

    private final String filePath;
    private final String datasetType;
    private final ErrorType errorType;
    private final int recordNumber;

    /**
     * Enumeration of common data loading error types
     */
    public enum ErrorType {
        FILE_NOT_FOUND("File not found"),
        FILE_NOT_READABLE("File not readable"),
        PARSE_ERROR("Parse error"),
        VALIDATION_ERROR("Validation error"),
        FORMAT_ERROR("Format error"),
        ENCODING_ERROR("Encoding error"),
        TIMEOUT_ERROR("Timeout error"),
        INSUFFICIENT_DATA("Insufficient data"),
        CORRUPT_DATA("Corrupt data"),
        UNKNOWN_ERROR("Unknown error");

        private final String displayName;

        ErrorType(String displayName) {
            this.displayName = displayName;
        }

        public String getDisplayName() {
            return displayName;
        }
    }

    /**
     * Basic constructor with message, file path, and dataset type
     */
    public DataLoadingException(String message, String filePath, String datasetType) {
        super(message);
        this.filePath = filePath;
        this.datasetType = datasetType;
        this.errorType = ErrorType.UNKNOWN_ERROR;
        this.recordNumber = -1;
    }

    /**
     * Constructor with cause
     */
    public DataLoadingException(String message, String filePath, String datasetType, Throwable cause) {
        super(message, cause);
        this.filePath = filePath;
        this.datasetType = datasetType;
        this.errorType = determineErrorTypeFromCause(cause);
        this.recordNumber = -1;
    }

    /**
     * Full constructor with all details
     */
    public DataLoadingException(String message, String filePath, String datasetType,
                                ErrorType errorType, int recordNumber) {
        super(message);
        this.filePath = filePath;
        this.datasetType = datasetType;
        this.errorType = errorType;
        this.recordNumber = recordNumber;
    }

    /**
     * Full constructor with cause
     */
    public DataLoadingException(String message, String filePath, String datasetType,
                                ErrorType errorType, int recordNumber, Throwable cause) {
        super(message, cause);
        this.filePath = filePath;
        this.datasetType = datasetType;
        this.errorType = errorType;
        this.recordNumber = recordNumber;
    }

    // Getters
    public String getFilePath() {
        return filePath;
    }

    public String getDatasetType() {
        return datasetType;
    }

    public ErrorType getErrorType() {
        return errorType;
    }

    public int getRecordNumber() {
        return recordNumber;
    }

    public boolean hasRecordNumber() {
        return recordNumber > 0;
    }

    /**
     * Determine error type from the exception cause
     */
    private ErrorType determineErrorTypeFromCause(Throwable cause) {
        if (cause == null) {
            return ErrorType.UNKNOWN_ERROR;
        }

        String causeType = cause.getClass().getSimpleName().toLowerCase();
        String message = cause.getMessage() != null ? cause.getMessage().toLowerCase() : "";

        if (causeType.contains("filenotfound") || message.contains("no such file")) {
            return ErrorType.FILE_NOT_FOUND;
        }
        if (causeType.contains("accessdenied") || message.contains("permission denied")) {
            return ErrorType.FILE_NOT_READABLE;
        }
        if (causeType.contains("unsupportedencoding") || message.contains("encoding")) {
            return ErrorType.ENCODING_ERROR;
        }
        if (causeType.contains("timeout")) {
            return ErrorType.TIMEOUT_ERROR;
        }
        if (causeType.contains("parse") || causeType.contains("json") || causeType.contains("csv")) {
            return ErrorType.PARSE_ERROR;
        }

        return ErrorType.UNKNOWN_ERROR;
    }

    @Override
    public String getMessage() {
        StringBuilder sb = new StringBuilder();

        // Add structured prefix
        sb.append("[").append(datasetType).append("]");

        if (errorType != ErrorType.UNKNOWN_ERROR) {
            sb.append(" ").append(errorType.getDisplayName()).append(":");
        }

        sb.append(" ");

        // Add main message
        sb.append(super.getMessage());

        // Add context information
        if (filePath != null && !filePath.isEmpty()) {
            sb.append(" (file: ").append(filePath).append(")");
        }

        if (hasRecordNumber()) {
            sb.append(" (record: ").append(recordNumber).append(")");
        }

        return sb.toString();
    }

    /**
     * Get a user-friendly error message without technical details
     */
    public String getUserFriendlyMessage() {
        switch (errorType) {
            case FILE_NOT_FOUND:
                return String.format("The file '%s' could not be found. Please check the file path.",
                        getFileName());
            case FILE_NOT_READABLE:
                return String.format("Cannot read the file '%s'. Please check file permissions.",
                        getFileName());
            case PARSE_ERROR:
                if (hasRecordNumber()) {
                    return String.format("There's a formatting error in '%s' at line %d. Please check the data format.",
                            getFileName(), recordNumber);
                } else {
                    return String.format("The file '%s' has formatting errors. Please check the data format.",
                            getFileName());
                }
            case VALIDATION_ERROR:
                return String.format("The data in '%s' doesn't match the expected format for %s.",
                        getFileName(), datasetType.toLowerCase());
            case INSUFFICIENT_DATA:
                return String.format("The file '%s' doesn't contain enough valid data to process.",
                        getFileName());
            case ENCODING_ERROR:
                return String.format("The file '%s' has character encoding issues. Try saving it as UTF-8.",
                        getFileName());
            default:
                return String.format("Failed to load %s from '%s': %s",
                        datasetType.toLowerCase(), getFileName(), super.getMessage());
        }
    }

    /**
     * Get just the filename from the full path
     */
    private String getFileName() {
        if (filePath == null || filePath.isEmpty()) {
            return "unknown file";
        }

        int lastSeparator = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'));
        if (lastSeparator >= 0 && lastSeparator < filePath.length() - 1) {
            return filePath.substring(lastSeparator + 1);
        }

        return filePath;
    }

    /**
     * Create detailed debug information
     */
    public String getDebugInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("DataLoadingException Debug Info:\n");
        sb.append("  Dataset Type: ").append(datasetType).append("\n");
        sb.append("  File Path: ").append(filePath).append("\n");
        sb.append("  Error Type: ").append(errorType.getDisplayName()).append("\n");
        sb.append("  Record Number: ").append(hasRecordNumber() ? recordNumber : "N/A").append("\n");
        sb.append("  Message: ").append(super.getMessage()).append("\n");

        if (getCause() != null) {
            sb.append("  Root Cause: ").append(getCause().getClass().getSimpleName()).append("\n");
            sb.append("  Root Message: ").append(getCause().getMessage()).append("\n");
        }

        return sb.toString();
    }

    /**
     * Check if this error is likely recoverable
     */
    public boolean isRecoverable() {
        return switch (errorType) {
            case FILE_NOT_FOUND, FILE_NOT_READABLE, FORMAT_ERROR, ENCODING_ERROR ->
                    false; // These require external fixes

            case PARSE_ERROR, VALIDATION_ERROR ->
                    hasRecordNumber(); // Row-level errors might be recoverable by skipping

            case TIMEOUT_ERROR -> true; // Can retry

            default -> false;
        };
    }

    // Static factory methods for common error scenarios

    public static DataLoadingException fileNotFound(String filePath, String datasetType) {
        return new DataLoadingException(
                "File does not exist: " + filePath,
                filePath,
                datasetType,
                ErrorType.FILE_NOT_FOUND,
                -1
        );
    }

    public static DataLoadingException fileNotReadable(String filePath, String datasetType) {
        return new DataLoadingException(
                "File is not readable: " + filePath,
                filePath,
                datasetType,
                ErrorType.FILE_NOT_READABLE,
                -1
        );
    }

    public static DataLoadingException parseError(String message, String filePath, String datasetType, int recordNumber) {
        return new DataLoadingException(
                message,
                filePath,
                datasetType,
                ErrorType.PARSE_ERROR,
                recordNumber
        );
    }

    public static DataLoadingException validationError(String message, String filePath, String datasetType) {
        return new DataLoadingException(
                message,
                filePath,
                datasetType,
                ErrorType.VALIDATION_ERROR,
                -1
        );
    }

    public static DataLoadingException insufficientData(String filePath, String datasetType, int recordsFound) {
        return new DataLoadingException(
                "Insufficient valid data found: " + recordsFound + " records",
                filePath,
                datasetType,
                ErrorType.INSUFFICIENT_DATA,
                recordsFound
        );
    }

    public static DataLoadingException corruptData(String message, String filePath, String datasetType) {
        return new DataLoadingException(
                "Data appears to be corrupt: " + message,
                filePath,
                datasetType,
                ErrorType.CORRUPT_DATA,
                -1
        );
    }

    @Override
    public String toString() {
        return String.format("DataLoadingException{type='%s', file='%s', error='%s', record=%s, message='%s'}",
                datasetType, getFileName(), errorType.getDisplayName(),
                hasRecordNumber() ? recordNumber : "N/A", super.getMessage());
    }
}