"""Create a sample PDF for testing the research agent."""
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

def create_sample_pdf():
    """Create a sample research paper PDF for testing."""
    
    # Create data/examples directory
    os.makedirs("./data/examples", exist_ok=True)
    
    # Create PDF
    filename = "./data/examples/sample_research.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Sample Research Paper: Statistical Analysis of Test Data")
    
    # Abstract
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 100, "Abstract")
    c.setFont("Helvetica", 10)
    
    abstract_text = [
        "This study examines the statistical significance of test data using various methodologies.",
        "We analyzed 120 participants with a mean age of 25.3 ± 4.2 years. The study employed",
        "t-tests and correlation analysis to determine relationships between variables.",
        "Results showed significant differences (p < 0.05) between experimental groups.",
    ]
    
    y_pos = height - 120
    for line in abstract_text:
        c.drawString(50, y_pos, line)
        y_pos -= 15
    
    # Methods
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos - 20, "Methods")
    c.setFont("Helvetica", 10)
    
    methods_text = [
        "Participants: 120 volunteers (60 male, 60 female)",
        "Age range: 18-35 years (M = 25.3, SD = 4.2)",
        "Design: Randomized controlled trial with two groups (n = 60 each)",
        "Statistical analysis: Independent t-tests, Pearson correlation",
        "Significance level: α = 0.05",
        "Software: Python with SciPy library"
    ]
    
    y_pos -= 40
    for line in methods_text:
        c.drawString(50, y_pos, line)
        y_pos -= 15
    
    # Results
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos - 20, "Results")
    c.setFont("Helvetica", 10)
    
    results_text = [
        "Primary analysis revealed significant differences between groups:",
        "- Group A: M = 78.5, SD = 12.3",
        "- Group B: M = 65.2, SD = 15.8", 
        "- t(118) = 5.23, p < 0.001, Cohen's d = 0.89",
        "",
        "Correlation analysis showed strong positive relationship:",
        "- Pearson's r = 0.73, p = 0.001, 95% CI [0.62, 0.81]",
        "- R² = 0.53, indicating 53% shared variance",
        "",
        "Secondary analysis with ANOVA:",
        "- F(2,117) = 12.45, p < 0.001, η² = 0.18"
    ]
    
    y_pos -= 40
    for line in results_text:
        c.drawString(50, y_pos, line)
        y_pos -= 15
    
    # Discussion
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos - 20, "Discussion")
    c.setFont("Helvetica", 10)
    
    discussion_text = [
        "The results demonstrate statistically significant effects with large effect sizes.",
        "The p-values below 0.05 indicate rejection of the null hypothesis.",
        "Effect sizes (Cohen's d = 0.89) suggest practical significance.",
        "Confidence intervals exclude zero, supporting the reliability of findings.",
        "These findings replicate previous research with similar methodologies."
    ]
    
    y_pos -= 40
    for line in discussion_text:
        c.drawString(50, y_pos, line)
        y_pos -= 15
    
    # Save the PDF
    c.save()
    print(f"✅ Sample PDF created: {filename}")
    return filename

if __name__ == "__main__":
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        pdf_path = create_sample_pdf()
        print(f"Sample research PDF ready for testing: {pdf_path}")
    except ImportError:
        print("❌ ReportLab not available. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "reportlab"])
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        pdf_path = create_sample_pdf()
        print(f"Sample research PDF ready for testing: {pdf_path}")