import glob
import os
from datetime import datetime, timedelta


def get_dated_directories(base_path):
    """
    Reads directories with YYYYMMDD structure from a given base path.
    """
    # Pattern to match YYYYMMDD: 4 digits, then 2 for month, then 2 for day
    pattern = os.path.join(base_path, "[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]")

    all_dirs = glob.glob(pattern)

    dated_dirs = []
    for d in all_dirs:
        if os.path.isdir(d):
            try:
                # Extract the YYYYMMDD part from the directory name
                dir_name = os.path.basename(d)
                date_str = dir_name  # Assuming the directory name is directly YYYYMMDD

                # Convert to datetime object
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                dated_dirs.append((date_obj, dir_name))
            except ValueError:
                # Skip directories that don't match the YYYYMMDD format
                continue

    # Sort the directories by date
    dated_dirs.sort(key=lambda x: x[0])
    return dated_dirs


def get_date_ranges(all_directories):
    list1_selected_weeks = []
    list2_skipped_weeks = []
    list3_aug_oct_2024 = []

    # Define the date ranges
    start_date_aug_2023 = datetime(2023, 8, 1)
    # The end date for the first two lists is August 1, 2024 (excluded)
    end_date_aug_2024_excluded = datetime(2024, 8, 1)

    # The start date for the third list is August 1, 2024
    start_date_aug_2024 = datetime(2024, 8, 1)
    end_date_oct_2024 = datetime(2024, 10, 30)

    # For the "3 weeks then skip 1" logic, we need to track weeks.
    # We'll use a `current_week_start` and a counter.

    # Filter directories relevant to the first two lists
    dirs_for_list1_2 = sorted(
        [
            (date_obj, dirname)
            for date_obj, dirname in all_directories
            if start_date_aug_2023 <= date_obj < end_date_aug_2024_excluded
        ]
    )

    # Process for List 1 and List 2 (August 2023 to August 2024 excluded)
    if dirs_for_list1_2:
        # Determine the Monday of the first relevant week
        first_date = dirs_for_list1_2[0][0]
        current_week_monday = first_date - timedelta(days=first_date.weekday())

        week_count = 0
        dates_in_current_week = []

        for date_obj, dirname in dirs_for_list1_2:
            # If the current date is past the current week's Monday + 7 days, it's a new week
            if date_obj >= current_week_monday + timedelta(weeks=1):
                # Process the previous week's dates
                if week_count % 4 < 3:  # 0, 1, 2 (first three weeks of the cycle)
                    list1_selected_weeks.extend(dates_in_current_week)
                else:  # 3 (fourth week of the cycle, which is skipped)
                    list2_skipped_weeks.extend(dates_in_current_week)

                # Reset for the new week
                current_week_monday = date_obj - timedelta(days=date_obj.weekday())
                week_count += 1
                dates_in_current_week = []

            dates_in_current_week.append(dirname)

        # Process any remaining dates from the last week
        if dates_in_current_week:
            if week_count % 4 < 3:
                list1_selected_weeks.extend(dates_in_current_week)
            else:
                list2_skipped_weeks.extend(dates_in_current_week)

    # Process for List 3 (August 1, 2024 to October 30, 2024)
    for date_obj, dirname in all_directories:
        if start_date_aug_2024 <= date_obj <= end_date_oct_2024:
            list3_aug_oct_2024.append(dirname)

    return list1_selected_weeks, list2_skipped_weeks, list3_aug_oct_2024


def get_train_val_test_dates(base_path):

    all_directories = get_dated_directories(base_path)
    return get_date_ranges(all_directories)
