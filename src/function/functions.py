from botocore.exceptions import NoCredentialsError
import boto3
import os
import pandas as pd


def download_data_from_s3(bucket_name, s3_folder, local_dir):
    """
    Downloads data from an S3 bucket to a local directory, and converts Parquet files to JSON.
    Args:
        bucket_name (str): The name of the S3 bucket.
        s3_folder (str): The folder path within the S3 bucket.
        local_dir (str): The local directory where files will be downloaded.
    Raises:
        NoCredentialsError: If AWS credentials are not available.
        Exception: For any other exceptions that occur during the download process.
    Returns:
        None
    """
    
    s3 = boto3.client('s3')
    
    try:
        # List objects within the specified S3 folder
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)
        
        if 'Contents' not in objects:
            print(f"No objects found in {s3_folder}")
            return
        
        for obj in objects['Contents']:
            s3_key = obj['Key']
            local_file_path = os.path.join(local_dir, os.path.relpath(s3_key, s3_folder))
            
            # Create local directory if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Download the file
            s3.download_file(bucket_name, s3_key, local_file_path)
            
            # Convert Parquet file to JSON
            parquet_to_json(local_file_path, local_file_path.replace('.parquet', '.json'))
            
            os.remove(local_file_path)
            
            print(f"Downloaded {s3_key} to {local_file_path}")
    
    except NoCredentialsError:
        print("Credentials not available")
    except Exception as e:
        print(f"An error occurred: {e}")


def parquet_to_json(parquet_file_path, json_file_path):
    """
    Converts a Parquet file to a JSON file.
    Parameters:
    parquet_file_path (str): The path to the input Parquet file.
    json_file_path (str): The path to the output JSON file.
    Returns:
    None
    This function reads a Parquet file into a pandas DataFrame, converts the DataFrame to a JSON string
    with records orientation and writes it to a specified JSON file. If an error occurs during the process,
    it prints an error message.
    """
    try:
        # Read the parquet file into a DataFrame
        df = pd.read_parquet(parquet_file_path)
        
        # Convert the DataFrame to a JSON string
        json_str = df.to_json(orient='records', lines=True)
        
        # Write the JSON string to a file
        with open(json_file_path, 'w') as json_file:
            json_file.write(json_str)
        
        print(f"Converted {parquet_file_path} to {json_file_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
