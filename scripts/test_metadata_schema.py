"""
Test script to directly register CSO metadata schema from YAML
"""
import yaml
import logging
from pathlib import Path
from core.metadata.registry import MetadataRegistry, MetadataSchema, MetadataField, MetadataFieldType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def main():
    try:
        # Load the YAML directly
        customer_id = "coupa"
        yaml_path = Path("/Users/ali.rafieefar/Documents/GitHub/ai_qna_assistant/customers/coupa/coupa.yaml")
        
        logger.info(f"Loading YAML from {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # Extract the dataset configuration
        ds_key = "cso_workflow_guides"
        ds_cfg = config.get("datasets", {}).get(ds_key, {})
        
        if not ds_cfg:
            logger.error(f"Dataset {ds_key} not found in YAML")
            return
            
        logger.info(f"Dataset config: {ds_cfg}")
        
        # Check for metadata schema
        metadata_schema = ds_cfg.get("metadata_schema")
        if not metadata_schema:
            logger.error(f"No metadata schema found for {ds_key}")
            return
            
        logger.info(f"Metadata schema: {metadata_schema}")
        
        # Register the schema
        schema_name = f"{customer_id}_{ds_key}"
        schema = MetadataSchema(
            name=schema_name,
            description=f"Schema for {customer_id} {ds_key} dataset"
        )
        
        # Add fields from the schema definition
        for field_name, field_def in metadata_schema.items():
            field_type_str = field_def.get("type", "string").lower()
            try:
                field_type = MetadataFieldType(field_type_str)
            except ValueError:
                logger.warning(f"Invalid field type '{field_type_str}' for field '{field_name}', defaulting to string")
                field_type = MetadataFieldType.STRING
                
            field = MetadataField(
                name=field_name,
                field_type=field_type,
                description=field_def.get("description", ""),
                required=field_def.get("required", False),
                default=field_def.get("default"),
                pattern=field_def.get("pattern")
            )
            
            schema.add_field(field)
        
        # Register the schema
        registry = MetadataRegistry()
        registry.register_schema(schema)
        logger.info(f"Successfully registered schema {schema_name}")
        
        # Check for sheets configuration
        sheet_cfg = ds_cfg.get("sheets", {})
        logger.info(f"Sheet config: {sheet_cfg}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
