import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def find_job(user_skills,top_n_jobs):
    job_database = pd.DataFrame({
        "job_role":["Backend Developer","Data Scientist","DevOps Engineer","Frontend Developer","Database Adminiastrator"],
        "Required_skills":[
            "Java Python SQL APIs Data Structures",
            "Python Machine Learning SQ Data Analysis TensorFlow",
            "AWS Docker Kubernetes Git",
            "HTML CSS JavaScript React",
            "SQL Database Management Optimization Oracle"]
    })
    
    user_profile = user_skills
    
    #appending user skills
    all_text_data = job_database["Required_skills"].tolist()
    all_text_data.append(user_profile)
    
    #make it as matrix
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(all_text_data)
    
    #dividing the database data and user input
    job_vectors = matrix[:-1]
    user_vector = matrix[-1]
    
    #comparing user input with database's data and finding the match score
    similarity_score = cosine_similarity(user_vector,job_vectors)[0]
    job_database["Match score"] = similarity_score
    
    #sorting according to match score
    sorted_jobs = job_database.sort_values(by="Match score",ascending=False)
    
    #finding top n jobs for the user skills
    final_recommendations = sorted_jobs.head(top_n_jobs)
    return final_recommendations

#User
skills = input("Enter the skills : ")
number = int(input("Enter number of jobs to find : " ))
career_paths = find_job(skills,number)
print("\nThe best career paths for your skills : \n")
for i,j in career_paths.iterrows():
    percentage_match = round(j["Match score"]*100,1)
    print(j["job_role"]," ~~ Percentage Match :",percentage_match,"\n")
    
    
    