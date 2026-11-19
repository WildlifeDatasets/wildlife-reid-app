def format_user_label(user):
    """Identify an account by name, email, and its unique username."""
    parts = [user.get_full_name().strip(), user.email, user.get_username()]
    return " — ".join(dict.fromkeys(part for part in parts if part))
