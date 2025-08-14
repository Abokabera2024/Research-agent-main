from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats, optimize
import re
import structlog

logger = structlog.get_logger()

def ttest_from_text(numbers_a: List[float], numbers_b: List[float]) -> Dict[str, Any]:
    """
    Perform independent t-test on two groups of numbers.
    
    Args:
        numbers_a: First group of numbers
        numbers_b: Second group of numbers
        
    Returns:
        Dictionary with test results
    """
    try:
        logger.debug("Performing t-test", 
                    group_a_size=len(numbers_a), 
                    group_b_size=len(numbers_b))
        
        if len(numbers_a) < 2 or len(numbers_b) < 2:
            return {
                "test": "t_independent", 
                "error": "Insufficient data for t-test",
                "min_required": 2
            }
        
        t_stat, p_value = stats.ttest_ind(
            numbers_a, 
            numbers_b, 
            equal_var=False, 
            nan_policy="omit"
        )
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(numbers_a) - 1) * np.var(numbers_a, ddof=1) + 
                             (len(numbers_b) - 1) * np.var(numbers_b, ddof=1)) / 
                            (len(numbers_a) + len(numbers_b) - 2))
        cohens_d = (np.mean(numbers_a) - np.mean(numbers_b)) / pooled_std
        
        result = {
            "test": "t_independent",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": bool(p_value < 0.05),
            "interpretation": "significant" if p_value < 0.05 else "not_significant",
            "sample_sizes": {"group_a": len(numbers_a), "group_b": len(numbers_b)}
        }
        
        logger.info("T-test completed", **result)
        return result
        
    except Exception as e:
        logger.error("T-test failed", error=str(e))
        return {"test": "t_independent", "error": str(e)}

def curve_fit_example(x: List[float], y: List[float]) -> Dict[str, Any]:
    """
    Perform linear curve fitting on x,y data.
    
    Args:
        x: Independent variable values
        y: Dependent variable values
        
    Returns:
        Dictionary with curve fitting results
    """
    try:
        logger.debug("Performing curve fitting", 
                    data_points=len(x))
        
        if len(x) != len(y) or len(x) < 3:
            return {
                "model": "linear", 
                "error": "Insufficient or mismatched data for curve fitting",
                "min_required": 3
            }
        
        # Linear model: y = a * x + b
        def linear_model(x, a, b):
            return a * x + b
        
        x_array = np.array(x)
        y_array = np.array(y)
        
        popt, pcov = optimize.curve_fit(linear_model, x_array, y_array)
        a, b = popt
        
        # Calculate R-squared
        y_pred = linear_model(x_array, a, b)
        ss_res = np.sum((y_array - y_pred) ** 2)
        ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate standard errors
        param_errors = np.sqrt(np.diag(pcov))
        
        result = {
            "model": "y = a*x + b",
            "parameters": {
                "a": float(a),
                "b": float(b)
            },
            "parameter_errors": {
                "a_error": float(param_errors[0]),
                "b_error": float(param_errors[1])
            },
            "r_squared": float(r_squared),
            "data_points": len(x),
            "quality": "good" if r_squared > 0.7 else "moderate" if r_squared > 0.4 else "poor"
        }
        
        logger.info("Curve fitting completed", **result)
        return result
        
    except Exception as e:
        logger.error("Curve fitting failed", error=str(e))
        return {"model": "linear", "error": str(e)}

def correlation_analysis(numbers_a: List[float], numbers_b: List[float]) -> Dict[str, Any]:
    """
    Perform correlation analysis between two variables.
    
    Args:
        numbers_a: First variable
        numbers_b: Second variable
        
    Returns:
        Dictionary with correlation results
    """
    try:
        logger.debug("Performing correlation analysis")
        
        if len(numbers_a) != len(numbers_b) or len(numbers_a) < 3:
            return {
                "test": "correlation", 
                "error": "Insufficient or mismatched data",
                "min_required": 3
            }
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(numbers_a, numbers_b)
        
        # Spearman correlation (non-parametric)
        spearman_r, spearman_p = stats.spearmanr(numbers_a, numbers_b)
        
        result = {
            "test": "correlation",
            "pearson": {
                "correlation": float(pearson_r),
                "p_value": float(pearson_p),
                "significant": bool(pearson_p < 0.05)
            },
            "spearman": {
                "correlation": float(spearman_r),
                "p_value": float(spearman_p),
                "significant": bool(spearman_p < 0.05)
            },
            "sample_size": len(numbers_a),
            "interpretation": {
                "strength": "strong" if abs(pearson_r) > 0.7 else "moderate" if abs(pearson_r) > 0.4 else "weak",
                "direction": "positive" if pearson_r > 0 else "negative"
            }
        }
        
        logger.info("Correlation analysis completed", **result)
        return result
        
    except Exception as e:
        logger.error("Correlation analysis failed", error=str(e))
        return {"test": "correlation", "error": str(e)}

def decide_need_scipy(text: str) -> bool:
    """
    Determine if SciPy analysis is needed based on text content.
    
    Args:
        text: Text content to analyze
        
    Returns:
        True if SciPy analysis is recommended, False otherwise
    """
    # Statistical terms that indicate need for analysis
    statistical_triggers = [
        "p-value", "p value", "p<", "p >", "p=",
        "regression", "correlation", "coefficient",
        "t-test", "t test", "anova", "chi-square",
        "significant", "significance", "confidence interval",
        "mean", "median", "standard deviation", "std",
        "hypothesis", "null hypothesis", "effect size",
        "statistical", "statistics", "analysis",
        "sample size", "n=", "df=", "degrees of freedom"
    ]
    
    # Mathematical patterns
    number_patterns = [
        r"p\s*[<>=]\s*\d+\.?\d*",  # p-values
        r"\d+\.?\d*\s*±\s*\d+\.?\d*",  # mean ± std
        r"r\s*=\s*\d+\.?\d*",  # correlation coefficients
        r"\(\d+,\s*\d+\)",  # sample sizes
        r"\d+\.?\d*\s*%",  # percentages
    ]
    
    text_lower = text.lower()
    
    # Check for statistical terms
    term_found = any(term in text_lower for term in statistical_triggers)
    
    # Check for numerical patterns
    pattern_found = any(re.search(pattern, text, re.IGNORECASE) for pattern in number_patterns)
    
    # Count numbers in the text
    numbers = re.findall(r"\d+\.?\d*", text)
    has_sufficient_numbers = len(numbers) >= 6  # Arbitrary threshold
    
    needs_scipy = term_found or pattern_found or has_sufficient_numbers
    
    logger.debug("SciPy need assessment", 
                needs_scipy=needs_scipy,
                term_found=term_found,
                pattern_found=pattern_found,
                number_count=len(numbers))
    
    return needs_scipy

def extract_numbers_from_text(text: str, max_numbers: int = 50) -> List[float]:
    """
    Extract numerical values from text.
    
    Args:
        text: Text to extract numbers from
        max_numbers: Maximum number of values to extract
        
    Returns:
        List of extracted numbers
    """
    try:
        # Pattern for numbers (including decimals and negative numbers)
        number_pattern = r"[-+]?\d*\.?\d+"
        matches = re.findall(number_pattern, text)
        
        numbers = []
        for match in matches[:max_numbers]:
            try:
                num = float(match)
                # Filter out unreasonable values
                if abs(num) < 1e10:  # Reasonable range
                    numbers.append(num)
            except ValueError:
                continue
        
        logger.debug("Extracted numbers from text", 
                    count=len(numbers), 
                    sample=numbers[:5] if numbers else [])
        
        return numbers
        
    except Exception as e:
        logger.error("Failed to extract numbers", error=str(e))
        return []

def comprehensive_analysis(text: str) -> Dict[str, Any]:
    """
    Perform comprehensive statistical analysis on text content.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Dictionary with all analysis results
    """
    try:
        logger.info("Starting comprehensive analysis")
        
        # Extract numbers
        numbers = extract_numbers_from_text(text)
        
        results = {
            "needs_scipy": decide_need_scipy(text),
            "numbers_extracted": len(numbers),
            "analysis_performed": []
        }
        
        if len(numbers) >= 4:
            # Split numbers into two groups for comparison
            mid = len(numbers) // 2
            group_a = numbers[:mid]
            group_b = numbers[mid:]
            
            # Perform t-test if we have enough data
            if len(group_a) >= 2 and len(group_b) >= 2:
                results["t_test"] = ttest_from_text(group_a, group_b)
                results["analysis_performed"].append("t_test")
            
            # Perform correlation analysis
            if len(numbers) >= 6:
                half = len(numbers) // 2
                x_vals = list(range(half))
                y_vals = numbers[:half]
                if len(x_vals) == len(y_vals):
                    results["correlation"] = correlation_analysis(x_vals, y_vals)
                    results["analysis_performed"].append("correlation")
            
            # Perform curve fitting
            if len(numbers) >= 6:
                x_vals = list(range(len(numbers)))
                results["curve_fit"] = curve_fit_example(x_vals, numbers)
                results["analysis_performed"].append("curve_fit")
        
        logger.info("Comprehensive analysis completed", 
                   analyses=results["analysis_performed"])
        
        return results
        
    except Exception as e:
        logger.error("Comprehensive analysis failed", error=str(e))
        return {
            "needs_scipy": False,
            "error": str(e),
            "analysis_performed": []
        }