def format_datetime(value, fmt='%Y년 %m월 %d일 %p %I:%M'):
    return value.strftime(fmt)

def calculate_total_page_cnt(total, per_page=20):
    return int(total//per_page) if total%per_page == 0 else int(total//per_page) + 1