"""June 2025. Cloning and archiving schemas and data for TRI 2025.

This script clones the current pipeline schemas into a new target database with prefix 'aeon_tri2025_'.
The cloned data will serve as a snapshot of the current state for TRI 2025 AEON Hackathon.
"""

import inspect
import os
from typing import Dict, List, Optional, Union

import datajoint as dj
from datajoint_utilities.dj_data_copy import db_migration
from datajoint_utilities.dj_data_copy.pipeline_cloning import ClonedPipeline

logger = dj.logger
os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"

# Database prefixes for source and target
SOURCE_DB_PREFIX: str = "aeon_"
TARGET_DB_PREFIX: str = "aeon_tri2025_"

# Map source schema names to target schema names
SCHEMA_NAME_MAPPER: Dict[str, str] = {
    SOURCE_DB_PREFIX + schema_name: TARGET_DB_PREFIX + schema_name
    for schema_name in (
        "lab",
        "subject",
        "acquisition",
        "tracking",
        "qc",
        "streams",
        "analysis",
        "block_analysis",
        "worker",
    )
}

# Restrict to specific experiments
RESTRICTION: List[Dict[str, str]] = [
    {"experiment_name": "social0.2-aeon3"},
    # {"experiment_name": "social0.2-aeon4"},
]

# Optional: Specify tables to exclude from cloning
TABLE_BLOCK_LIST: Dict[str, List[str]] = {
    f"{TARGET_DB_PREFIX}acquisition": ["Chunk.File"],
    f"{TARGET_DB_PREFIX}streams": ["SpinnakerVideoSourceVideo", "WeightScaleWeightRaw"],
    f"{TARGET_DB_PREFIX}tracking": ["SLEAPTracking.Part"],
    f"{TARGET_DB_PREFIX}qc": ["CameraQC"],
}

# Optional: Batch size for data migration (None for no batching)
BATCH_SIZE: Optional[int] = 10


def clone_pipeline() -> None:
    """Clone the pipeline structure into the target database.
    
    Creates a new set of schemas with the target prefix, maintaining the same
    table structure and relationships as the source schemas.
    """
    diagram = None
    for orig_schema_name in SCHEMA_NAME_MAPPER:
        virtual_module = dj.create_virtual_module(orig_schema_name, orig_schema_name)
        if diagram is None:
            diagram = dj.Diagram(virtual_module)
        else:
            diagram += dj.Diagram(virtual_module)

    cloned_pipeline = ClonedPipeline(diagram, SCHEMA_NAME_MAPPER, verbose=True)
    cloned_pipeline.instantiate_pipeline(prompt=False)


def data_copy(
    restriction: Optional[List[Dict[str, str]]] = None,
    table_block_list: Dict[str, List[str]] = None,
    batch_size: Optional[int] = None,
) -> None:
    """Migrate data from source to target schemas.
    
    Args:
        restriction: Optional list of restrictions to filter data
        table_block_list: Optional dict mapping schema names to lists of tables to exclude
        batch_size: Optional batch size for data migration
    """
    for orig_schema_name, cloned_schema_name in SCHEMA_NAME_MAPPER.items():
        orig_schema = dj.create_virtual_module(orig_schema_name, orig_schema_name)
        cloned_schema = dj.create_virtual_module(cloned_schema_name, cloned_schema_name)

        db_migration.migrate_schema(
            orig_schema,
            cloned_schema,
            restriction=restriction,
            table_block_list=table_block_list.get(cloned_schema_name, []),
            allow_missing_destination_tables=True,
            force_fetch=bool(batch_size),  # Force fetch if batch_size is specified,
            batch_size=batch_size,
        )


def validate() -> Dict[str, Union[List[str], Dict[str, List[str]], Dict[str, Dict[str, Dict[str, int]]]]]:
    """Validate the schema migration process.
    
    Performs three levels of validation:
    1. Checks if all schemas have been migrated
    2. Verifies all tables within each schema have been migrated
    3. Compares entry counts and database sizes for each table
    
    Returns:
        Dict containing validation results with:
        - missing_schemas: List of schemas that failed to migrate
        - missing_tables: Dict mapping schemas to their missing tables
        - missing_entries: Dict with entry count and size differences
    """
    missing_schemas: List[str] = []
    missing_tables: Dict[str, List[str]] = {}
    missing_entries: Dict[str, Dict[str, Dict[str, int]]] = {}

    for orig_schema_name, cloned_schema_name in SCHEMA_NAME_MAPPER.items():
        logger.info(f"Validating schema: {orig_schema_name}")
        source_vm = dj.create_virtual_module(orig_schema_name, orig_schema_name)

        try:
            target_vm = dj.create_virtual_module(cloned_schema_name, cloned_schema_name)
        except dj.errors.DataJointError:
            missing_schemas.append(orig_schema_name)
            continue

        missing_tables[orig_schema_name] = []
        missing_entries[orig_schema_name] = {}

        for attr in dir(source_vm):
            obj = getattr(source_vm, attr)
            if isinstance(obj, dj.user_tables.UserTable) or (
                inspect.isclass(obj) and issubclass(obj, dj.user_tables.UserTable)
            ):
                source_tbl = obj
                try:
                    target_tbl = getattr(target_vm, attr)
                except AttributeError:
                    missing_tables[orig_schema_name].append(source_tbl.table_name)
                    continue
                
                logger.info(f"\tValidating entry count: {source_tbl.__name__}")
                source_entry_count = len(source_tbl())
                target_entry_count = len(target_tbl())
                missing_entries[orig_schema_name][source_tbl.__name__] = {
                    "entry_count_diff": source_entry_count - target_entry_count,
                    "db_size_diff": source_tbl().size_on_disk - target_tbl().size_on_disk,
                }

    return {
        "missing_schemas": missing_schemas,
        "missing_tables": missing_tables,
        "missing_entries": missing_entries,
    }


if __name__ == "__main__":
    # Clone the pipeline structure
    clone_pipeline()
    
    # Copy the data
    data_copy(
        restriction=RESTRICTION,
        table_block_list=TABLE_BLOCK_LIST,
        batch_size=BATCH_SIZE,
    )
    
    # Validate the migration
    validation_results = validate()
    
    # Print validation summary
    print("\nValidation Summary:")
    print(f"Missing schemas: {validation_results['missing_schemas']}")
    print("\nMissing tables by schema:")
    for schema, tables in validation_results['missing_tables'].items():
        if tables:
            print(f"{schema}: {tables}")
    print("\nEntry count differences by table:")
    for schema, tables in validation_results['missing_entries'].items():
        for table, diffs in tables.items():
            if diffs['entry_count_diff'] != 0 or diffs['db_size_diff'] != 0:
                print(f"{schema}.{table}: {diffs}") 