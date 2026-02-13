import pdfplumber
import re

MODULE_PATTERN = re.compile(
    r'Module\s*:\s*(\d+)\s+(.+?)\s+(\d+)\s*hours?',
    re.IGNORECASE
)

def extract_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def crop_syllabus_section(text):
    start = text.find("Module:")
    end = text.find("Total Lecture hours")

    if start != -1 and end != -1:
        return text[start:end]

    return text

def split_into_modules(text):
    matches = list(MODULE_PATTERN.finditer(text))
    modules = []

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)

        modules.append({
            "module_no": match.group(1),
            "module_name": match.group(2).strip(),
            "hours": int(match.group(3)),
            "content": text[start:end].strip()
        })

    return modules

def extract_topics(module_text):
    text = module_text.replace("\n", " ")
    text = text.replace(" – ", " - ")
    text = text.replace(", and", ", ")

    protected = ""
    bracket = False
    for ch in text:
        if ch == "(":
            bracket = True
        if ch == ")":
            bracket = False

        if ch == "-" and not bracket:
            protected += " $$$ "
        else:
            protected += ch

    raw_topics = protected.split("$$$")

    topics = []
    for t in raw_topics:
        t = t.strip(" :,-")
        if len(t) > 3:
            topics.append(t)

    return topics


def parse_syllabus_pdf(pdf_file):
    text = extract_pdf_text(pdf_file)
    text = crop_syllabus_section(text)

    modules = split_into_modules(text)

    parsed = []
    for mod in modules:
        parsed.append({
            "module_name": mod["module_name"],
            "hours": mod["hours"],
            "topics": extract_topics(mod["content"])
        })

    return parsed[:7]

