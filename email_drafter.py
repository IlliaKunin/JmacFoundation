"""Build the JakeyMac Foundation email with a supplied recipient name."""

import argparse


EMAIL_TEMPLATE = """Hi [First Name],
I'm reaching out from JakeyMac Foundation. We are developing SmartFast Learning, a program focused on helping young people build confidence, improve focus, and develop real-world skills in a world increasingly influenced by technology.
Our goal is much bigger than running a few local programs. We want to bring SmartFast Learning to schools and communities across the United States, reaching thousands of young people and building something that can make a real difference in how the next generation learns and grows.
We are now looking to connect with a few potential founding partners who share this vision and are interested in helping us turn it into something much bigger.
I would be happy to share more about our plans, how we want to scale the program, and the impact we hope to achieve.
Would you be open to a short conversation?
Best regards,
Jim McArthur
JakeyMac Foundation"""


def draft_email(name: str) -> str:
    """Insert the supplied name verbatim, trimming surrounding whitespace."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A recipient name is required.")
    name = name.strip()
    if "\n" in name or "\r" in name:
        raise ValueError("The recipient name must fit on one line.")
    return EMAIL_TEMPLATE.replace("[First Name]", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Recipient name to insert in the greeting.")
    args = parser.parse_args()
    try:
        email = draft_email(args.name)
    except ValueError as error:
        parser.error(str(error))
    print(email)


if __name__ == "__main__":
    main()
