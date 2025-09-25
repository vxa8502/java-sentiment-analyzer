package sentiment.data;

/**
 * Simplified exception for data loading errors.
 * Focuses only on essential information without complex categorization.
 */
public class DataLoadingException extends Exception {

    private final String filePath;
    private final String datasetType;

    /**
     * Basic constructor with message, file path, and dataset type
     */
    public DataLoadingException(String message, String filePath, String datasetType) {
        super(message);
        this.filePath = filePath;
        this.datasetType = datasetType;
    }

    /**
     * Constructor with cause
     */
    public DataLoadingException(String message, String filePath, String datasetType, Throwable cause) {
        super(message, cause);
        this.filePath = filePath;
        this.datasetType = datasetType;
    }

    // Getters
    public String getFilePath() {
        return filePath;
    }

    public String getDatasetType() {
        return datasetType;
    }

    @Override
    public String getMessage() {
        StringBuilder sb = new StringBuilder();

        // Add structured prefix
        sb.append("[").append(datasetType).append("] ");

        // Add main message
        sb.append(super.getMessage());

        // Add context information
        if (filePath != null && !filePath.isEmpty()) {
            sb.append(" (file: ").append(getFileName()).append(")");
        }

        return sb.toString();
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

    @Override
    public String toString() {
        return String.format("DataLoadingException{type='%s', file='%s', message='%s'}",
                datasetType, getFileName(), super.getMessage());
    }
}