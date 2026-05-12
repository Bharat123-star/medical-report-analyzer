# medical-report-analyzer
this is my medical repository
<br>
By Bharat trivedi 
Athrav Tiwari 
Aryan Shukla
Prashant Shukla
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in .env file")

if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY is missing in .env file")


# =========================================================
# PDF CONFIGURATION
# =========================================================

PDF_PATH = "WM17S.pdf"

# Pages start from 1
PAGES_TO_EXTRACT = [1, 3]


# =========================================================
# FUNCTION TO EXTRACT PDF TEXT
# =========================================================

def extract_pdf_text(pdf_path, pages):
    """
    Extract text safely from selected PDF pages
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    extracted_text = ""

    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)

            total_pages = len(reader.pages)

            for page_num in pages:

                # Validate page number
                if page_num < 1 or page_num > total_pages:
                    print(f"Skipping invalid page number: {page_num}")
                    continue

                page = reader.pages[page_num - 1]

                text = page.extract_text()

                if text:
                    extracted_text += text + "\n"

    except Exception as e:
        raise Exception(f"Error while reading PDF: {str(e)}")

    if not extracted_text.strip():
        raise ValueError("No text could be extracted from PDF")

    # Prevent huge prompt size
    return extracted_text[:6000]


# =========================================================
# EXTRACT TEXT
# =========================================================

raw_text = extract_pdf_text(PDF_PATH, PAGES_TO_EXTRACT)

print("\n========== EXTRACTED PDF TEXT ==========\n")
print(raw_text)
print("\n========================================\n")


# =========================================================
# LLM CONFIGURATION
# =========================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)


# =========================================================
# TOOLS
# =========================================================

search_tool = SerperDevTool()


# =========================================================
# AGENTS
# =========================================================

blood_test_analyst = Agent(
    role="Blood Test Analyst",
    goal="Analyze blood test reports accurately and summarize abnormalities.",
    backstory=(
        "You are an experienced pathology expert specializing "
        "in blood test interpretation and clinical summaries."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

article_researcher = Agent(
    role="Medical Research Researcher",
    goal="Find reliable medical articles related to blood test abnormalities.",
    backstory=(
        "You are a professional medical researcher skilled in "
        "finding authentic healthcare resources and studies."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

health_advisor = Agent(
    role="Health Advisor",
    goal="Provide general wellness recommendations based on findings.",
    backstory=(
        "You are a wellness advisor who gives safe, general "
        "health recommendations based on medical research."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)


# =========================================================
# TASKS
# =========================================================

analyze_blood_test_task = Task(
    description=f"""
    Analyze the following blood test report carefully.

    BLOOD REPORT:
    {raw_text}

    Identify:
    - Abnormal values
    - Possible concerns
    - Important observations
    - Summary of overall health indicators
    """,

    expected_output="""
    A detailed but concise analysis of the blood report including:
    - Abnormal findings
    - Medical interpretation
    - Overall summary
    """,

    agent=blood_test_analyst
)

find_articles_task = Task(
    description="""
    Based on the blood test analysis,
    search for reliable medical articles and health resources.

    Prioritize:
    - Mayo Clinic
    - WebMD
    - Healthline
    - WHO
    - NIH

    Include article titles and links.
    """,

    expected_output="""
    A list of reliable medical articles with:
    - Article title
    - Short summary
    - Website link
    """,

    agent=article_researcher,
    context=[analyze_blood_test_task]
)

provide_recommendations_task = Task(
    description="""
    Based on the blood report analysis and researched articles,
    provide general health and wellness recommendations.

    Do NOT diagnose diseases.
    Do NOT prescribe medicines.
    Give only general lifestyle guidance.
    """,

    expected_output="""
    General wellness recommendations including:
    - Diet suggestions
    - Exercise suggestions
    - Hydration advice
    - Sleep recommendations
    - When to consult a doctor
    """,

    agent=health_advisor,
    context=[find_articles_task]
)


# =========================================================
# CREATE CREW
# =========================================================

crew = Crew(
    agents=[
        blood_test_analyst,
        article_researcher,
        health_advisor
    ],

    tasks=[
        analyze_blood_test_task,
        find_articles_task,
        provide_recommendations_task
    ],

    process=Process.sequential,
    verbose=True
)


# =========================================================
# EXECUTE CREW
# =========================================================

if __name__ == "__main__":

    try:
        print("\n========== RUNNING CREW ==========\n")

        result = crew.kickoff()

        print("\n========== FINAL RESULT ==========\n")
        print(result)

    except Exception as e:
        print("\n========== ERROR ==========\n")
        print(str(e))
