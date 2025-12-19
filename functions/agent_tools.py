"""
Here you will find the tools given to the Agent, specially to read and write
in the Local Files.
"""

from langchain.tools import tool
from pathlib import Path
from docx import Document
from .logger import log_tool_call


@tool(
    "read_folder",
    parse_docstring=True,
    description="reads the existing folder names with client names"
)
def read_folder(path : str) -> str:
    """
    Description:
        Read the folder as a list of the names on 1 level.

    Args:
        path (str): this is the path where the folders you are searching for are.

    returns:
        The exact and correct names of the clients

    raises:
        Error if the path is empty

    """
    log_tool_call("read_folder", {"path": path})
    active_path = Path(path)
    clients = [folder.name for folder in active_path.iterdir() if folder.is_dir()]
    log_tool_call("read_folder", {"path": path}, output=str(clients), status="success")

    return clients

@tool(
    "read_template",
    parse_docstring=True,
    description="reads the desired Template with desired structure"
)
def read_template(path : str, template_name : str) -> str:
    """
    Description:
        Reads the desired template structure.

    Args:
        path (str): this is the path where the template documents are.
        template_name (str): the desired template to read.

    returns:
        The template content to understand the structure inside it

    raises:
        Error if the template is not readable or does not exist

    """
    log_tool_call("read_template", {"path": path, "template_name": template_name})
    final_path = f"{path}/{template_name}"

    file_path = Path(final_path)
    doc = Document(file_path)

    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    log_tool_call("read_template", {"template_name": template_name}, output=f"Loaded {len(text)} chars", status="success")

    return text

@tool(
    "verify_document_draft",
    parse_docstring=True,
    description="MUST use before saving any document. Checks if draft meets character limits."
)
def verify_document_draft(content: str, document_type: str) -> str:
    """
    Description:
        Verifies that a document draft meets the character limits before saving.
        You MUST call this tool before using any save tool.

    Args:
        content (str): The full document draft to verify
        document_type (str): Type of document - "summary", "homework", or "draft"

    Returns:
        Analysis showing character count per section, limits, and PASS/FAIL status.
        If any section fails, you must edit the draft and verify again before saving.
    """
    log_tool_call("verify_document_draft", {"document_type": document_type, "content_length": len(content)})
    lines = content.split('\n')
    total_chars = len(content)

    if document_type == "summary":
        # (min, max) ranges - sweet spot is middle of range
        limits = {
            "total": (2200, 2500),           # sweet spot: 2300
            "title": (15, 35),               # sweet spot: 25
            "warm_opening": (180, 300),      # sweet spot: 240
            "main_takeaways": (700, 1000),   # sweet spot: 850
            "tools": (150, 300),             # sweet spot: 225
            "achievements": (280, 500),      # sweet spot: 390
            "next_steps": (120, 200)         # sweet spot: 160
        }

        # Parse sections
        title = lines[0] if lines else ""
        warm_opening = lines[2] if len(lines) > 2 else ""

        sections = {}
        current_section = None
        current_content = []
        section_markers = {
            "Main Takeaways": "main_takeaways",
            "Core \"Why\"": "core_why",
            "Tools": "tools",
            "Most Recent Achievements!": "achievements",
            "Most Recent Achievements": "achievements",
            "Next Steps": "next_steps"
        }

        for line in lines[3:]:
            if line.strip() in section_markers:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = section_markers[line.strip()]
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        # Build result
        results = []
        all_pass = True

        def check_range(value, min_val, max_val):
            if value < min_val:
                return f"✗ TOO SHORT (need {min_val - value} more)"
            elif value > max_val:
                return f"✗ TOO LONG (remove {value - max_val})"
            else:
                return "✓ PASS"

        # Total check
        total_min, total_max = limits['total']
        total_status = check_range(total_chars, total_min, total_max)
        results.append(f"TOTAL: {total_chars} chars (target: {total_min}-{total_max}) {total_status}")
        if total_chars < total_min or total_chars > total_max:
            all_pass = False

        # Title check
        title_len = len(title)
        title_min, title_max = limits['title']
        title_status = check_range(title_len, title_min, title_max)
        results.append(f"Title: {title_len} chars (target: {title_min}-{title_max}) {title_status}")
        if title_len < title_min or title_len > title_max:
            all_pass = False

        # Warm opening check
        warm_len = len(warm_opening)
        warm_min, warm_max = limits['warm_opening']
        warm_status = check_range(warm_len, warm_min, warm_max)
        results.append(f"Warm Opening: {warm_len} chars (target: {warm_min}-{warm_max}) {warm_status}")
        if warm_len < warm_min or warm_len > warm_max:
            all_pass = False

        # Section checks
        for section_key in ["main_takeaways", "tools", "achievements", "next_steps"]:
            section_content = sections.get(section_key, "")
            section_len = len(section_content)
            min_val, max_val = limits[section_key]
            status = check_range(section_len, min_val, max_val)
            passed = min_val <= section_len <= max_val
            if not passed:
                all_pass = False
            section_name = section_key.replace("_", " ").title()
            results.append(f"{section_name}: {section_len} chars (target: {min_val}-{max_val}) {status}")

        results.append("")
        if all_pass:
            results.append("=== ALL SECTIONS PASS === You may now save the document.")
            log_tool_call("verify_document_draft", {"document_type": document_type, "total_chars": total_chars}, output="ALL PASS", status="success")
        else:
            results.append("=== SOME SECTIONS FAIL === Adjust content to fit target ranges, then verify again.")
            log_tool_call("verify_document_draft", {"document_type": document_type, "total_chars": total_chars}, output="FAILED", status="fail")

        return '\n'.join(results)

    elif document_type == "homework":
        if total_chars <= 1800:
            log_tool_call("verify_document_draft", {"document_type": "homework"}, output=f"PASS ({total_chars}/1800)", status="success")
            return f"TOTAL: {total_chars} / 1800 chars ✓ PASS\n\n=== DOCUMENT PASSES === You may now save."
        else:
            log_tool_call("verify_document_draft", {"document_type": "homework"}, output=f"FAIL ({total_chars}/1800)", status="fail")
            return f"TOTAL: {total_chars} / 1800 chars ✗ FAIL - OVER BY {total_chars - 1800}\n\n=== DOCUMENT FAILS === Edit to reduce characters, then verify again."

    elif document_type == "draft":
        if total_chars <= 2500:
            log_tool_call("verify_document_draft", {"document_type": "draft"}, output=f"PASS ({total_chars}/2500)", status="success")
            return f"TOTAL: {total_chars} / 2500 chars ✓ PASS\n\n=== DOCUMENT PASSES === You may now save."
        else:
            log_tool_call("verify_document_draft", {"document_type": "draft"}, output=f"FAIL ({total_chars}/2500)", status="fail")
            return f"TOTAL: {total_chars} / 2500 chars ✗ FAIL - OVER BY {total_chars - 2500}\n\n=== DOCUMENT FAILS === Edit to reduce characters, then verify again."

    else:
        log_tool_call("verify_document_draft", {"document_type": document_type}, output="Unknown type", status="error")
        return f"Unknown document type: {document_type}. Use 'summary', 'homework', or 'draft'."


@tool(
    "save_summary",
    parse_docstring=True,
    description="saves the summary of the transcription and user notes in the correct Session Path"
)
def save_summary(path: str, content : str) -> str:
    """
    Description:
        saves the created summary in the current Session Path.

    Args:
        path (str): The current Session Path given in the context
        content (str): The summary created following the summary_template guidelines

    Returns:
        Confirmation that the summary has been saved in the Session Path

    Raises:
        Error is the path is incorrect of was an issue while saving.
    """
    log_tool_call("save_summary", {"path": path, "content_length": len(content)})
    file_path = Path(f"{path}/summary.txt")
    file_path.write_text(content, encoding="utf-8")
    log_tool_call("save_summary", {"path": path}, output=str(file_path), status="saved")

    return f"new summary created in {file_path}"

@tool(
    "save_homework",
    parse_docstring=True,
    description="saves the homework aligned with the transcription and user notes in the correct Session Path"
)
def save_homework(path: str, content : str) -> str:
    """
    Description:
        saves the created homework in the current Session Path.

    Args:
        path (str): The current Session Path given in the context
        content (str): The homework created following the homework_template guidelines

    Returns:
        Confirmation that the homework has been saved in the Session Path

    Raises:
        Error is the path is incorrect of was an issue while saving.
    """
    log_tool_call("save_homework", {"path": path, "content_length": len(content)})
    file_path = Path(f"{path}/homework.txt")
    file_path.write_text(content, encoding="utf-8")
    log_tool_call("save_homework", {"path": path}, output=str(file_path), status="saved")

    return f"new homework created in {file_path}"

@tool(
    "save_next_session_draft",
    parse_docstring=True,
    description="saves the next session draft aligned with the transcription and user notes in the correct Session Path"
)
def save_session_draft(path: str, content : str) -> str:
    """
    Description:
        saves the created next session draft in the current Session Path.

    Args:
        path (str): The current Session Path given in the context
        content (str): The session draft created following the next_session_template guidelines

    Returns:
        Confirmation that the session draft has been saved in the Session Path

    Raises:
        Error is the path is incorrect of was an issue while saving.
    """
    log_tool_call("save_next_session_draft", {"path": path, "content_length": len(content)})
    file_path = Path(f"{path}/next_session.txt")
    file_path.write_text(content, encoding="utf-8")
    log_tool_call("save_next_session_draft", {"path": path}, output=str(file_path), status="saved")

    return f"new next session draft created in {file_path}"