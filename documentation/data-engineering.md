# Data Engineer

Responsible for building and maintaining the data architecture of a data science project => ability to design and build data warehouses.

## Core Skills
* Introduction to Data Engineering
* Python, Java, C++
* Operating Systems
* Databases: SQL, noSQL
* Data Warehouses: Hadoop, MapReduce, Hive, Pig, Apache Spark, Kafka
* Basic Machine Learning

## Resources

1. [A Beginner’s Guide to Data Engineering — Part I](https://medium.com/@rchang/a-beginners-guide-to-data-engineering-part-i-4227c5c457d7)
A data warehouse is a place where raw data is transformed and stored in query-able forms.

ETL Pipeline

**Extract:** Sensors wait for upstream data sources to land (e.g. an upstream source could be a mahcine or user-generated logs, relational database copy, external dataset, etc.). Upon available, we transport the data from their source locations to further transformations.

**Transform:** Application of business logic and perform actions such as filtering, grouping and aggregation to translate raw data into analysis-ready datasets. This step requires a great deal of business understanding and domain knowledge.

**Load:** We load the processed data and transport them to a final destination. This dataset can be either consumed directly by end-users or it can be treated as yet another upstream dependency on another ETL job, forming the so called [data lineage](https://en.wikipedia.org/wiki/Data_lineage).

Important features of frameworks:

**Configuration:** We need to be able to succinctly describe the data flow of a data pipeline. Is the ETL configured on a UI, a domain specific language, or code? Code allows users to expressively build pipelines programmatically that are customizable.

**UI, Monitoring, Alerts:** Monitoring and Alerts are crucial in tracking the progress of long running processes. How well does a framework provide visual information for job progress? Does it surface alerts or warnings in a timely and accurate manner?

**Backfilling:** Once a data pipeline is built, we often need to go back in time and re-process the historical data. 


2. .
3. .
