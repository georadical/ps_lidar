import sys
from pathlib import Path
from src.core.io import PointCloudLoader

def main():
    # Default test file found in external_references
    default_file = Path("external_references/3DFin_Tutorial/Data/plot_1.laz")
    
    # Allow command line override
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        file_path = default_file

    print(f"--- Brick 1 Verification: Loading {file_path} ---")

    if not file_path.exists():
        print(f"ERROR: Test file not found at {file_path}")
        print("Please provide a path to a .las/.laz file: python test_brick_1.py <path>")
        return

    try:
        loader = PointCloudLoader(file_path)
        loader.load()
        meta = loader.get_metadata()
        
        print("\nSUCCESS! Metadata extracted:")
        for key, value in meta.items():
            print(f"{key}: {value}")
            
        print(f"\nCoordinates shape: {loader.get_xyz().shape}")
        
    except Exception as e:
        print(f"\nFAILURE: An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
