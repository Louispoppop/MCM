import pandas as pd

# Load data
df = pd.read_csv("Cleaned_data_with_votes.csv")

# Clean industry names first (handle case sensitivity)
df["celebrity_industry"] = df["celebrity_industry"].str.strip().str.title()


# Map
def get_category(ind):
    if pd.isna(ind):
        return "Other"

    ind = ind.strip()

    # 1. High Physicality (Athletic)
    if ind in [
        "Athlete",
        "Racing Driver",
        "Fitness Instructor",
        "Astronaut",
        "Military",
    ]:
        return "Athletic"

    # 2. Performance & Art
    if ind in [
        "Actor/Actress",
        "Singer/Rapper",
        "Musician",
        "Comedian",
        "Model",
        "Magician",
        "Fashion Designer",
        "Beauty Pagent",
        "Beauty Pageant",
    ]:
        return "Performance"

    # 3. Media & TV
    # Note: 'Tv Personality' -> Title case makes it 'Tv Personality' or 'Tv personality'?
    # Let's just check loosely
    if ind in [
        "Tv Personality",
        "News Anchor",
        "Sports Broadcaster",
        "Radio Personality",
        "Journalist",
        "Social Media Personality",
        "Motivational Speaker",
    ]:
        return "Media"

    return "Other"


df["industry_group"] = df["celebrity_industry"].apply(get_category)

print("Classification Summary:")
for group in ["Athletic", "Performance", "Media", "Other"]:
    print(f"\n[{group}]")
    occupations = sorted(
        df[df["industry_group"] == group]["celebrity_industry"].unique()
    )
    print(", ".join(occupations))
