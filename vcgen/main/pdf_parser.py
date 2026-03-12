import pdfplumber
import re
 
MODULE_PATTERN = re.compile(
    r'Module\s*:\s*(\d+)\s+(.+?)\s+(\d+)\s*hours?',
    re.IGNORECASE
)
 
JUNK_PATTERNS = [
    re.compile(r'Agenda\s+Item', re.IGNORECASE),
    re.compile(r'Proceedings\s+of', re.IGNORECASE),
    re.compile(r'Annexure', re.IGNORECASE),
    re.compile(r'Academic\s+Council', re.IGNORECASE),
    re.compile(r'^\s*\d+\s*$'),         
    re.compile(r'Board\s+of\s+Studies', re.IGNORECASE),
    re.compile(r'Approved\s+by', re.IGNORECASE),
    re.compile(r'Recommended\s+by', re.IGNORECASE),
    re.compile(r'Mode\s+of\s+Evaluation', re.IGNORECASE),
    re.compile(r'Text\s+Book', re.IGNORECASE),
    re.compile(r'Reference\s+Book', re.IGNORECASE),
]
 
 
def _is_junk_line(line: str) -> bool:
    return any(p.search(line) for p in JUNK_PATTERNS)
 
 
def extract_pdf_text(pdf_file) -> str:
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            raw = page.extract_text()
            if not raw:
                continue
            clean_lines = [ln for ln in raw.splitlines() if not _is_junk_line(ln)]
            text += "\n".join(clean_lines) + "\n"
    return text
 
 
def _join_continuation_lines(text: str) -> str:

    lines = text.splitlines()
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r'Module\s*:\s*\d+', line, re.IGNORECASE) and \
                not re.search(r'\d+\s*hours?', line, re.IGNORECASE):
            if i + 1 < len(lines) and not re.match(r'Module\s*:', lines[i+1], re.IGNORECASE):
                line = line.rstrip() + " " + lines[i+1].strip()
                i += 1
        result.append(line)
        i += 1
    return "\n".join(result)
 
 
def crop_syllabus_section(text: str) -> str:
    start = text.find("Module:")
    end = text.find("Total Lecture hours")
 
    if start != -1 and end != -1:
        return text[start:end]
    return text
 
 
def split_into_modules(text: str) -> list:
    text = _join_continuation_lines(text)
 
    matches = list(MODULE_PATTERN.finditer(text))
    modules = []
 
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
 
        raw_name = match.group(2).strip()
        module_name = re.sub(r'\s+', ' ', raw_name)
 
        modules.append({
            "module_no":   match.group(1),
            "module_name": module_name,
            "hours":       int(match.group(3)),
            "content":     text[start:end].strip(),
        })
 
    return modules
 
 
def extract_topics(module_text: str) -> list:
    text = module_text.replace("\n", " ")
 
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace(" – ", " - ").replace(" — ", " - ")
 
    text = text.replace(", and ", ", ")
 
    parts = []
    depth = 0
    current = []
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            current.append(ch)
        elif depth == 0 and text[idx:idx+3] == ' - ':
            parts.append(''.join(current))
            current = []
            idx += 3  
            continue
        else:
            current.append(ch)
        idx += 1
    if current:
        parts.append(''.join(current))
 
    topics = []
    for t in parts:
        t = t.strip(" :,")
        if len(t) > 4:
            topics.append(t)
 
    return topics
 
 
def parse_syllabus_pdf(pdf_file) -> list:
    text = extract_pdf_text(pdf_file)
    text = crop_syllabus_section(text)
 
    modules = split_into_modules(text)
 
    parsed = []
    for mod in modules:
        parsed.append({
            "module_name": mod["module_name"],
            "hours":       mod["hours"],
            "topics":      extract_topics(mod["content"]),
        })
 
    return parsed[:7]
 