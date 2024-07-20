# Advanced fraud detection for bank and e-commerce
# Data 
Ip_to_country which has lower and upperd ip_address with their corresponding country 

fraud_data for e-commerce, that has all the necessary dataframes for us to make best predicitons, including source, browser, ip_address, device_id, signup_time, purchase_time, Amount, class and others. When class 1 shows fraudulent and class 0 non_fraudulent

credit_data for bank transaction, data frames are Time, V1-V8, Amount and class, where V1-V8 has no description for it is used for security purpose.

# This project will involve:

 Analyzing and preprocessing transaction data.
 Creating and engineering features that help identify fraud patterns.
 Building and training machine learning models to detect fraud.
 Evaluating model performance and making necessary improvements.
 Deploying the models for real-time fraud detection and setting up monitoring for continuous improvement.


# Knowledge:
 Principles of model deployment and serving
 Best practices for creating REST APIs
 Understanding of containerization and its benefits
 Techniques for real-time prediction serving
 Security considerations in API development
 Methods for monitoring and maintaining deployed models



# Usage
# pull docker image 
 sudo docker pull fraud_detection for linux and mac
 docker pull fraud_detection
 # check if it is exist 
 sudo docker ps linux or mac/ docke ps windows
 REPOSITORY            TAG         IMAGE ID       CREATED             SIZE
fraud_detection  
 # build the docker image 
 sudo docker run -p 5000:5000 fraud_detection 


 feel free to modify for any issue contact me at 
 # yadasat437@gmail.com