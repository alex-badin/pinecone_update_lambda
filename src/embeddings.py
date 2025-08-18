# src/embeddings.py (Revised for vertexai SDK)

import json
import os
import time
import uuid
import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

# Use the higher-level vertexai library
import vertexai
from vertexai.language_models import TextEmbeddingModel # Use the specific class
from google.cloud import storage
from google.cloud.aiplatform import compat # For job state constants if needed
# Import job state enums specifically if needed, otherwise rely on vertexai handling
# from google.cloud.aiplatform_v1.types.job_state import JobState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Embedder:
    def __init__(self, project_id: str, location: str = "us-central1", staging_bucket: str = None, cache_db=None, batch_size=100):
        """
        Initializes the Embedder using the vertexai SDK.

        Args:
            project_id: Your Google Cloud project ID.
            location: The GCP region for Vertex AI operations (e.g., "us-central1").
            staging_bucket: GCS bucket name for staging input/output files (e.g., "your-bucket-name").
                              Must be in the same region as the Vertex AI job.
            cache_db: Path to a potential cache DB (currently unused).
            batch_size: Parameter kept for signature consistency, but batching is largely
                        handled by the batch prediction service itself based on the input file size.
                        May influence local file preparation if input list is huge.
        """
        if not project_id:
            raise ValueError("Google Cloud Project ID is required.")
        if not staging_bucket:
            raise ValueError("GCS staging bucket name is required for batch predictions.")

        self.project_id = project_id
        self.location = location
        self.staging_bucket_name = staging_bucket # Just the bucket name
        # Define a base path within the bucket for organization
        self.gcs_staging_base = f"gs://{staging_bucket}/embedding_batch_staging_vertexai_sdk"
        self.cache_db = cache_db # Currently unused
        self.batch_size = batch_size # Max items if pre-chunking large lists locally

        # Initialize Vertex AI SDK globally
        try:
            vertexai.init(project=project_id, location=location, staging_bucket=f"gs://{staging_bucket}")
            self.storage_client = storage.Client(project=project_id)
             # Instantiate the specific embedding model - adjust model name as needed
            self.model = TextEmbeddingModel.from_pretrained("text-embedding-004") # Or "textembedding-gecko@003", etc.

        except Exception as e:
            logging.error(f"Failed to initialize Vertex AI SDK or Storage client: {e}")
            raise

        logging.info(f"Embedder initialized for project '{project_id}' in location '{location}' using bucket '{staging_bucket}' with model '{self.model._model_name}'.")


    def _upload_to_gcs(self, source_file_name: str, destination_blob_name: str) -> str:
        """Uploads a local file to the GCS staging bucket."""
        try:
            bucket = self.storage_client.bucket(self.staging_bucket_name)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_name)
            gcs_uri = f"gs://{self.staging_bucket_name}/{destination_blob_name}"
            logging.info(f"File {source_file_name} uploaded to {gcs_uri}")
            return gcs_uri
        except Exception as e:
            logging.error(f"Failed to upload {source_file_name} to GCS: {e}")
            raise

    def _prepare_batch_input_file(self, texts: List[str], local_file_path: str):
        """
        Creates a JSONL file where each line is a JSON object expected by the model.
        Example format for text-embedding models often requires 'content'.
        """
        with open(local_file_path, 'w', encoding='utf-8') as f:
            for text_content in texts:
                # Format expected by many text embedding models via batch_predict
                # Includes 'content' and optionally 'task_type' or 'title'
                instance = {
                    "content": text_content,
                    "task_type": "RETRIEVAL_DOCUMENT", # Adjust task_type based on your use case and model
                    "title": "" # Optional: Provide title if relevant
                 }
                json_line = json.dumps(instance)
                f.write(json_line + '\n')
        logging.info(f"Prepared batch input file: {local_file_path} with {len(texts)} items.")

    def _download_blob(self, blob: storage.Blob, destination_file_name: str):
        """Downloads a blob from GCS."""
        try:
            os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
            blob.download_to_filename(destination_file_name)
            logging.info(f"Downloaded GCS file {blob.name} to {destination_file_name}")
        except Exception as e:
            logging.error(f"Failed to download GCS file {blob.name}: {e}")
            raise

    def _parse_batch_output(self, gcs_output_uri: str, local_download_dir: str) -> List[Optional[List[float]]]:
        """
        Downloads and parses the JSONL output files from GCS, assuming order is preserved.

        Returns:
            A list of embedding vectors (list[float]) or None for failures,
            in the same order as the input texts.
        """
        if not gcs_output_uri.startswith("gs://"):
             logging.error(f"Invalid GCS output URI: {gcs_output_uri}")
             return []

        bucket_name = gcs_output_uri.split('/')[2]
        # Handle potential trailing slash in prefix
        prefix = '/'.join(gcs_output_uri.split('/')[3:])
        if prefix and not prefix.endswith('/'):
             prefix += '/'

        bucket = self.storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix)) # List all files

        if not blobs:
            logging.warning(f"No output blobs found at prefix: {gcs_output_uri}")
            return []

        embeddings_list: List[Optional[List[float]]] = []
        os.makedirs(local_download_dir, exist_ok=True)

        # Sort blobs to process them in a predictable order (important!)
        # Vertex AI often names them like prediction-<model>-<job_id>-<shard_id>.jsonl
        blobs.sort(key=lambda b: b.name)

        processed_lines = 0
        for blob in blobs:
            # Output files are typically named like 'predictions_*.jsonl'
            if blob.name.endswith('.jsonl') and 'prediction' in blob.name.lower():
                local_file_path = os.path.join(local_download_dir, os.path.basename(blob.name))
                logging.info(f"Processing result file: {blob.name}")
                self._download_blob(blob, local_file_path)

                # Parse the downloaded JSONL file line by line
                with open(local_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            result = json.loads(line.strip())
                            # The exact structure depends on the model version.
                            # Common structure: {"instance": {input_fields...}, "prediction": {"embeddings": {"values": [...]}}}
                            # Or sometimes directly: {"embeddings": {"values": [...]}} if input isn't echoed back.
                            # Adapt this extraction logic based on actual output format.
                            if 'prediction' in result and 'embeddings' in result['prediction']:
                                embedding = result['prediction']['embeddings'].get('values')
                            elif 'embeddings' in result: # Simpler format
                                embedding = result['embeddings'].get('values')
                            else:
                                embedding = None
                                logging.warning(f"Could not find 'embeddings' structure in line: {line.strip()}")

                            if isinstance(embedding, list):
                                embeddings_list.append(embedding)
                            else:
                                embeddings_list.append(None) # Append None if embedding is invalid or missing
                                logging.warning(f"Invalid or missing embedding values in line: {line.strip()}")

                        except json.JSONDecodeError:
                            logging.warning(f"Skipping invalid JSON line in {blob.name}: {line.strip()}")
                            embeddings_list.append(None) # Append None for parse errors
                        except Exception as e:
                             logging.error(f"Unexpected error processing line in {blob.name}: {e} - Line: {line.strip()}")
                             embeddings_list.append(None) # Append None for other errors
                        processed_lines += 1

                # Optionally remove the local file after processing
                # os.remove(local_file_path)

        logging.info(f"Parsed {len(embeddings_list)} embedding results from {gcs_output_uri} ({processed_lines} lines processed).")
        return embeddings_list

    async def run_batch_embedding_job(self, texts: List[str],
                                        job_display_name_prefix: str = "vertexai-sdk-batch-embedding") -> List[Optional[List[float]]]:
        """
        Prepares data, runs a Vertex AI Batch Prediction job using the vertexai SDK,
        waits for completion, and returns the results IN ORDER.

        Args:
            texts: A list of strings to embed.
            job_display_name_prefix: A prefix for the Vertex AI job display name.

        Returns:
            A list containing the embedding vector (list[float]) for each input text,
            or None if embedding failed for a specific text. The order matches the input 'texts'.
            Returns an empty list if the job fails entirely before producing results.
        """
        if not texts:
            logging.warning("No texts provided for embedding.")
            return []

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_suffix = f"{timestamp}-{uuid.uuid4().hex[:8]}"
        job_display_name = f"{job_display_name_prefix}-{unique_suffix}"

        # --- 1. Prepare and Upload Input Data ---
        local_input_filename = f"batch_input_{unique_suffix}.jsonl"
        gcs_input_uri = None
        try:
            self._prepare_batch_input_file(texts, local_input_filename)

            # Define GCS path using the base staging path and unique suffix
            gcs_input_blob_name = f"{self.gcs_staging_base.split('//')[1].split('/', 1)[1]}/input/{local_input_filename}" # Path within bucket
            gcs_input_uri = self._upload_to_gcs(local_input_filename, gcs_input_blob_name)
        finally:
            # Clean up local input file
            if os.path.exists(local_input_filename):
                try:
                    os.remove(local_input_filename)
                except OSError as e:
                    logging.warning(f"Could not delete local input file {local_input_filename}: {e}")

        if not gcs_input_uri:
            logging.error("Failed to prepare or upload input file to GCS. Aborting job.")
            return []

        # --- 2. Configure and Submit Batch Prediction Job ---
        # Output URI prefix within the staging base path
        gcs_output_uri_prefix = f"{self.gcs_staging_base}/{job_display_name}/output/"

        job = None
        try:
            logging.info(f"Submitting batch prediction job '{job_display_name}'...")
            # Use the model's batch_predict method
            job = self.model.batch_predict(
                instances_format="jsonl",       # Format of the input file
                predictions_format="jsonl",    # Desired format for the output file
                gcs_source=gcs_input_uri,        # GCS path to the input file
                gcs_destination_prefix=gcs_output_uri_prefix, # GCS path for output folder
                job_display_name=job_display_name,
                # Optional: sync=False makes the call non-blocking initially, but we'll wait below.
                # sync=True, # Or handle waiting manually below
                # Optional: model_parameters, machine_type etc can often be passed here too
                # model_parameters = {} # e.g. {"task_type": "RETRIEVAL_DOCUMENT"} if needed explicitly
            )

            # The returned 'job' object is typically an instance of aiplatform.BatchPredictionJob
            logging.info(f"Submitted Batch Prediction Job: {job.resource_name}")
            logging.info(f"View Job Status: https://console.cloud.google.com/vertex-ai/locations/{self.location}/batch-predictions/{job.name}?project={self.project_id}")
            logging.info(f"Job State after submission: {job.state.name}")


            # --- 3. Wait for Job Completion ---
            logging.info("Waiting for job to complete...")
            # job.wait() # Simple synchronous wait (can block for a long time)

            # Asynchronous polling wait (more responsive in async contexts)
            polling_interval_seconds = 60 # Check every minute
            max_wait_minutes = 120 # Timeout after 2 hours (adjust as needed)
            start_time = time.time()

            while job.state not in [
                # Using JobState enum members directly for clarity
                vertexai.aiplatform.compat.types.job_state.JobState.JOB_STATE_SUCCEEDED,
                vertexai.aiplatform.compat.types.job_state.JobState.JOB_STATE_FAILED,
                vertexai.aiplatform.compat.types.job_state.JobState.JOB_STATE_CANCELLED,
                vertexai.aiplatform.compat.types.job_state.JobState.JOB_STATE_EXPIRED,
                vertexai.aiplatform.compat.types.job_state.JobState.JOB_STATE_PARTIALLY_SUCCEEDED, # Treat as success for parsing
            ]:
                if time.time() - start_time > max_wait_minutes * 60:
                     logging.error(f"Job timed out after {max_wait_minutes} minutes.")
                     # Optional: Try to cancel the job
                     # job.cancel()
                     return [] # Return empty on timeout

                await asyncio.sleep(polling_interval_seconds)
                job.refresh() # Update job state from the API
                logging.info(f"Job state: {job.state.name}...")


            logging.info(f"Job {job.resource_name} finished with state: {job.state.name}")

            # --- 4. Process Results ---
            # Treat PARTIALLY_SUCCEEDED the same as SUCCEEDED for result parsing
            if job.state in [
                 vertexai.aiplatform.compat.types.job_state.JobState.JOB_STATE_SUCCEEDED,
                 vertexai.aiplatform.compat.types.job_state.JobState.JOB_STATE_PARTIALLY_SUCCEEDED
                 ]:
                logging.info("Job finished successfully (or partially). Downloading and parsing results...")
                local_download_dir = f"batch_output_{unique_suffix}"
                embeddings_list: List[Optional[List[float]]] = []
                try:
                    # Use the GCS output path from the completed job object
                    output_gcs_path = job.output_info.gcs_output_directory
                    embeddings_list = self._parse_batch_output(output_gcs_path, local_download_dir)
                    return embeddings_list
                finally:
                    # Clean up local download directory
                    if os.path.exists(local_download_dir):
                        import shutil
                        try:
                            shutil.rmtree(local_download_dir)
                        except OSError as e:
                             logging.warning(f"Could not delete local output directory {local_download_dir}: {e}")
                    # Optional: Clean up GCS staging files (input/output)
                    # self._delete_gcs_directory(gcs_input_uri)
                    # self._delete_gcs_directory(job.output_info.gcs_output_directory)

            else:
                logging.error(f"Batch prediction job failed, was cancelled, or expired. State: {job.state.name}")
                if job.error:
                     logging.error(f"Job Error: {job.error.message}") # Access error message if available
                return [] # Return empty list on failure

        except Exception as e:
             logging.exception(f"An error occurred during batch job submission or processing: {e}")
             # If job object exists and has failed, log its state
             if job:
                 job.refresh()
                 logging.error(f"Final job state before exception: {job.state.name}")
                 if job.error:
                     logging.error(f"Job Error: {job.error.message}")
             return [] # Return empty list on unexpected error


    # Placeholder for potential GCS cleanup utility
    def _delete_gcs_directory(self, gcs_path: str):
        """Deletes all blobs under a given GCS prefix."""
        if not gcs_path.startswith("gs://"):
            logging.warning(f"Invalid GCS path for deletion: {gcs_path}")
            return
        try:
            bucket_name = gcs_path.split('/')[2]
            prefix = '/'.join(gcs_path.split('/')[3:])
            if prefix and not prefix.endswith('/'): # Ensure prefix ends with / if it's meant to be a directory
                 prefix += '/'

            bucket = self.storage_client.bucket(bucket_name)
            blobs_to_delete = list(bucket.list_blobs(prefix=prefix))
            if blobs_to_delete:
                 # GCS client library can delete blobs in batches for efficiency
                 # bucket.delete_blobs(blobs_to_delete) # Check API for batch deletion
                 # Or delete one by one:
                 for blob in blobs_to_delete:
                     blob.delete()
                 logging.info(f"Deleted {len(blobs_to_delete)} blobs from GCS path: {gcs_path}")
            else:
                 logging.info(f"No blobs found to delete at GCS path: {gcs_path}")

        except Exception as e:
             logging.error(f"Failed to delete GCS directory {gcs_path}: {e}")