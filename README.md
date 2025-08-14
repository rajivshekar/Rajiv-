Cybersecurity: Suspicious Web Threat Interactions
**1. Project Overview**
This project focuses on detecting and analyzing suspicious web traffic patterns to identify potentially malicious activities. The analysis uses a dataset of web traffic records collected through AWS CloudWatch to a production web server. The methodology includes data preprocessing, exploratory data analysis (EDA), feature engineering, and training a machine learning model to classify traffic as suspicious.
**2. Dataset**
The dataset used is 

CloudWatch_Traffic_Web_Attack.csv. It contains web traffic records that have been labeled as suspicious by various detection rules. Each record represents a stream of traffic to a web server  and includes the following key columns:

bytes_in: Bytes received by the server.

bytes_out: Bytes sent from the server.

src_ip: The source IP address.

src_ip_country_code: The country code of the source IP.

protocol: The protocol used in the connection.

response.code: The HTTP response code.

dst_port: The destination port on the server.

rule_names: The name of the rule that flagged the traffic as suspicious.

detection_types: The type of detection applied.

**3. Analysis and Methodology**
The project follows a systematic approach to analyze and model the data:

Data Preprocessing: The initial step involved loading the dataset, checking for missing values, and converting time-related columns (creation_time, end_time) to the correct datetime format.

Exploratory Data Analysis (EDA): The data was analyzed to understand traffic patterns. This included visualizing the distribution of incoming and outgoing bytes and counting the frequency of traffic by source country code.

Feature Engineering: New features were created from existing data to aid in the analysis. A key feature added was 

session_duration, which calculates the duration of the connection in seconds based on the creation_time and end_time.

Modeling: The dataset is ideal for building classification or anomaly detection models. A Random Forest Classifier was used to predict whether a session is suspicious, and an Isolation Forest model was employed for anomaly detection.

Evaluation: The Random Forest Classifier was evaluated using a classification report to determine its precision, recall, and accuracy in identifying suspicious activities.

**4. Key Findings**
The analysis provided several insights into the nature of the suspicious traffic:

Data Flow: High bytes_in and low bytes_out sessions could indicate possible infiltration attempts.

Geographic Source: Frequent interactions from specific country codes may indicate targeted or bot-related attacks.

Port Activity: High activity on non-standard ports may signal unauthorized access attempts. In this dataset, all traffic records were on the standard port 

443.

**5. Technologies Used**
Python: The primary programming language.

pandas: For data manipulation and analysis.

scikit-learn: For data preprocessing and machine learning modeling.

matplotlib & seaborn: For data visualization.

Git: For version control and repository management.
