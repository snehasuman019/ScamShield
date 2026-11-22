# ScamShield Dataset Directory

This directory stores the datasets used by ScamShield for training machine learning models and populating MongoDB Atlas.

## Primary Dataset: EMSCAD (Employment Scam Aegean Dataset)

- **Filename**: `fake_job_postings.csv`
- **Total Records**: 17,880 genuine job postings published between 2012 and 2014.
- **Target Label Column**: `fraudulent`
  - `0`: Legitimate job posting (17,014 records, ~95.16%)
  - `1`: Fraudulent / Scam job posting (866 records, ~4.84%)
- **Source**: University of the Aegean / Kaggle Real or Fake Job Postings dataset (`shivamb/real-or-fake-fake-jobposting-prediction`).

## Columns Specification

1. `job_id`: Unique identifier for each job ad.
2. `title`: Job posting title.
3. `location`: Geographical location in format `Country, State, City`.
4. `department`: Corporate department (e.g. Sales, Engineering).
5. `salary_range`: Range if disclosed (e.g. `50000-75000`).
6. `company_profile`: Overview of the recruiting company.
7. `description`: Detailed text explaining the role, responsibilities, and tasks.
8. `requirements`: Requisite qualifications, degrees, experience, and skills.
9. `benefits`: Perks, healthcare, PTO, bonuses.
10. `telecommuting`: Binary flag (0 = On-site, 1 = Remote allowed).
11. `has_company_logo`: Binary flag (0 = No logo, 1 = Company logo present).
12. `has_questions`: Binary flag (0 = No screening questions, 1 = Screening questions present).
13. `employment_type`: Full-time, Part-time, Contract, Temporary, Other.
14. `required_experience`: Entry level, Associate, Mid-Senior, Executive, Internship, Not Applicable.
15. `required_education`: Bachelor's Degree, High School or equivalent, Master's Degree, etc.
16. `industry`: Industry domain (e.g. Information Technology, Healthcare).
17. `function`: Professional job function (e.g. Engineering, Sales).
18. `fraudulent`: Target ground truth label (0 = Legitimate, 1 = Fraudulent).

## Automatic Download

Run the following command from the project root:

```bash
python data/download_dataset.py
```
