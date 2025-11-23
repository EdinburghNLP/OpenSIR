import re


def is_math_expression(s: str) -> bool:
    """
    Determine if a string represents a mathematical expression using general patterns.
    """
    # special case
    if s.strip() == "\\boxed{ANSWER}":
        return False

    # Skip empty strings
    if not s or s.isspace():
        return False

    # Filter out LaTeX text commands (non-mathematical text)
    if re.match(r"\\text\{[^}]*\}", s):
        return False

    # Accept pure numbers (including decimals and negative numbers)
    if re.match(r"^-?\d+(\.\d+)?$", s):
        return True

    # Accept decimal numbers starting with a dot (.35625)
    if re.match(r"^-?\.\d+$", s):
        return True

    # Accept numbers with thousands separators (12,500.00)
    if re.match(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$", s):
        return True

    # Accept comma-separated number sequences (15, 21, 27)
    if re.match(r"^\d+(\.\d+)?(,\s*\d+(\.\d+)?)+$", s):
        return True

    # Accept ratio notation (4:5, 3:2:1)
    if re.match(r"^\d+(\.\d+)?(:\d+(\.\d+)?)+$", s):
        return True

    # Accept percentages (0%, 25.5%, 0\%)
    if re.match(r"^\d+(\.\d+)?\\?%$", s):
        return True

    # Accept factorials (14!, 5!)
    if re.match(r"^\d+!$", s):
        return True

    # Accept currency notation with escaped dollar signs (\$18.90, \$36)
    if re.match(r"^\\\$\d+(\.\d+)?$", s):
        return True

    # Accept currency notation (£500, €250.50, $142.86, ¥1000, 142.86$)
    currency_symbols = r"[\$£€¥¢₹₽₪₨₩₦₡₵₴₸₺₼₽]"
    if re.match(
        rf"^({currency_symbols}?\d+(\.\d+)?{currency_symbols}?|\d+(\.\d+)?{currency_symbols})$",
        s,
    ):
        return True

    # Accept fractions
    if re.match(r"^\d+/\d+$", s):
        return True

    # Accept mathematical expressions with LaTeX symbols
    latex_math_patterns = [
        r"\\(pi|theta|alpha|beta|gamma|delta|epsilon|sigma|mu|nu|rho|tau|phi|psi|omega)",
        r"\\(int|sum|prod|lim|sqrt|frac|sin|cos|tan|log|ln|exp)",
        r"\\(circ|infty)",
        r"\\;",  # LaTeX spacing
        r"\\\w+",  # General LaTeX commands
    ]

    for pattern in latex_math_patterns:
        if re.search(pattern, s):
            return True

    # Accept expressions with mathematical operators
    if re.search(r"[\+\-\*/\^=<>(){}[\]]", s):
        return True

    # Accept mathematical notation patterns
    math_notation_patterns = [
        r"^\d+\^\d+$",  # powers like 135^2
        r"^\d+[a-z]$",  # variables like 2a
        r"^\d+\$$",  # numbers with $ like 1$
        r"^[a-zA-Z]{1,2}$",  # single/double letter variables
        r"[_\^]",  # subscripts/superscripts
        r"E_[a-z]",  # scientific notation with subscripts
        r"^\d+(\.\d+)?[eE][+-]?\d+$",  # scientific notation (1.23e-4)
    ]

    for pattern in math_notation_patterns:
        if re.search(pattern, s):
            return True

    # Filter out strings that are clearly descriptive text
    # Long strings with multiple words and spaces are likely descriptions
    if (
        len(s) > 10
        and " " in s
        and not re.search(r"[\d\+\-\*/\^=<>(){}[\]\\%!$:£€¥¢₹₽₪₨₩₦₡₵₴₸₺₼]", s)
    ):
        return False

    # Filter out all-caps strings that are likely labels/tokens
    if s.isupper() and len(s) > 3 and s.isalpha():
        return False

    # Filter out strings that are purely alphabetic and longer than 3 characters
    # (likely to be words rather than mathematical variables)
    if re.match(r"^[A-Za-z\s]+$", s) and len(s) > 3:
        return False

    # Accept short alphanumeric strings (likely variables/constants)
    if len(s) <= 3 and re.match(r"^[A-Za-z0-9]+$", s):
        return True

    # Accept strings with mixed numbers and letters (mathematical expressions)
    if re.search(r"\d", s) and re.search(r"[a-zA-Z]", s):
        return True

    # Default: reject if none of the above patterns match
    return False
