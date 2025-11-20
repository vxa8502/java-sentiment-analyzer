package sentiment.models;

import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import java.util.Objects;

/**
 * SVM configuration for hyperparameter tuning and model training.
 */
public class SVMConfig {

    public enum KernelType {
        LINEAR("Linear", "weka.classifiers.functions.supportVector.PolyKernel"),
        POLYNOMIAL("Polynomial", "weka.classifiers.functions.supportVector.PolyKernel"),
        RBF("RBF", "weka.classifiers.functions.supportVector.RBFKernel");

        private final String displayName;
        private final String wekaClassName;

        KernelType(String displayName, String wekaClassName) {
            this.displayName = displayName;
            this.wekaClassName = wekaClassName;
        }

        public String getDisplayName() {
            return displayName;
        }

        public String getWekaClassName() {
            return wekaClassName;
        }
    }

    private final double c;
    private final KernelType kernelType;
    private final double gamma;  // For RBF kernel
    private final int degree;    // For polynomial kernel
    private final double[] classWeights;  // null means no class weighting
    private final double epsilon;

    // Performance metrics from cross-validation
    private Double cvMacroF1;
    private Double cvAccuracy;
    private Double cvStdDev;

    /**
     * Creates an SVM configuration with linear kernel.
     */
    public SVMConfig(double c) {
        this(c, KernelType.LINEAR, 0.01, 1, null, 1.0E-12);
    }

    /**
     * Creates an SVM configuration.
     *
     * @param c complexity parameter
     * @param kernelType kernel type
     * @param gamma RBF kernel parameter
     * @param degree polynomial kernel degree
     * @param classWeights class weights, null for none
     * @param epsilon tolerance parameter
     */
    public SVMConfig(double c, KernelType kernelType, double gamma, int degree,
                     double[] classWeights, double epsilon) {
        if (c <= 0) {
            throw new IllegalArgumentException("C must be positive");
        }
        if (gamma <= 0) {
            throw new IllegalArgumentException("Gamma must be positive");
        }
        if (degree < 1) {
            throw new IllegalArgumentException("Degree must be >= 1");
        }
        if (epsilon <= 0) {
            throw new IllegalArgumentException("Epsilon must be positive");
        }

        this.c = c;
        this.kernelType = kernelType;
        this.gamma = gamma;
        this.degree = degree;
        this.classWeights = classWeights != null ? classWeights.clone() : null;
        this.epsilon = epsilon;
    }

    /**
     * Creates a kernel instance.
     */
    public Kernel createKernel() {
        switch (kernelType) {
            case LINEAR:
                PolyKernel linearKernel = new PolyKernel();
                linearKernel.setExponent(1.0);  // Linear = polynomial with degree 1
                return linearKernel;

            case POLYNOMIAL:
                PolyKernel polyKernel = new PolyKernel();
                polyKernel.setExponent(degree);
                return polyKernel;

            case RBF:
                RBFKernel rbfKernel = new RBFKernel();
                rbfKernel.setGamma(gamma);
                return rbfKernel;

            default:
                throw new IllegalStateException("Unknown kernel type: " + kernelType);
        }
    }

    /**
     * Builds options string for SMO.
     */
    public String toOptionsString() {
        return "-C " + c +
               " -L 0.001" +  // Learning rate
               " -P " + epsilon +
               " -N 0" +  // Filter type
               " -V -1" + // Number of folds for cross-validation calibration
               " -W 1";   // Kernel cache size
    }

    // Getters

    public double getC() {
        return c;
    }

    public KernelType getKernelType() {
        return kernelType;
    }

    public double getGamma() {
        return gamma;
    }

    public int getDegree() {
        return degree;
    }

    public double[] getClassWeights() {
        return classWeights != null ? classWeights.clone() : null;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public Double getCvMacroF1() {
        return cvMacroF1;
    }

    public Double getCvAccuracy() {
        return cvAccuracy;
    }

    public Double getCvStdDev() {
        return cvStdDev;
    }

    // Setters for CV results (package-private, only HyperparameterSearch should use)

    void setCvMacroF1(double cvMacroF1) {
        this.cvMacroF1 = cvMacroF1;
    }

    void setCvAccuracy(double cvAccuracy) {
        this.cvAccuracy = cvAccuracy;
    }

    void setCvStdDev(double cvStdDev) {
        this.cvStdDev = cvStdDev;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("SVMConfig{");
        sb.append("C=").append(c);
        sb.append(", kernel=").append(kernelType.getDisplayName());

        if (kernelType == KernelType.RBF) {
            sb.append(", gamma=").append(gamma);
        } else if (kernelType == KernelType.POLYNOMIAL) {
            sb.append(", degree=").append(degree);
        }

        if (classWeights != null) {
            sb.append(", weighted=true");
        }

        if (cvMacroF1 != null) {
            sb.append(", CV_F1=").append(String.format("%.4f", cvMacroF1));
        }
        if (cvAccuracy != null) {
            sb.append(", CV_Acc=").append(String.format("%.4f", cvAccuracy));
        }
        if (cvStdDev != null) {
            sb.append(", CV_StdDev=").append(String.format("%.4f", cvStdDev));
        }

        sb.append("}");
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SVMConfig that = (SVMConfig) o;
        return Double.compare(that.c, c) == 0 &&
               Double.compare(that.gamma, gamma) == 0 &&
               degree == that.degree &&
               Double.compare(that.epsilon, epsilon) == 0 &&
               kernelType == that.kernelType;
    }

    @Override
    public int hashCode() {
        return Objects.hash(c, kernelType, gamma, degree, epsilon);
    }
}
