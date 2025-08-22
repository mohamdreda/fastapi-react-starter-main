# Data Transformation Module

This module provides functionality for transforming data through encoding and scaling operations. It's designed to be modular, allowing you to easily add new transformation methods as needed.

## Features

### Categorical Encoding
- **One-Hot Encoding**: Convert categorical variables into a one-hot numeric array.
- **Label Encoding**: Encode target labels with value between 0 and n_classes-1.

### Feature Scaling
- **Robust Scaling**: Scale features using statistics that are robust to outliers.
- **Standard Scaling**: Standardize features by removing the mean and scaling to unit variance.

## Usage

### Initialization

```python
from app.services.data_transformation.transformation_service import DataTransformationService

# Initialize the service
service = DataTransformationService(
    dataset_id=1,  # ID of the dataset
    user_id=1,     # ID of the user performing the transformation
    base_path=None # Optional base path for saving artifacts
)
```

### Configuration

Create a configuration dictionary to specify the transformations to apply:

```python
config = {
    'categorical_encoding': {
        'methods': [
            {
                'method': 'one_hot',
                'columns': ['category_column'],
                'drop': 'first'  # Optional: 'first', 'if_binary', or None
            },
            {
                'method': 'label',
                'columns': ['ordinal_column']
            }
        ]
    },
    'feature_scaling': {
        'methods': [
            {
                'method': 'robust',
                'columns': ['numeric_column_with_outliers'],
                'with_centering': True,  # Optional
                'with_scaling': True,     # Optional
                'quantile_range': (25.0, 75.0)  # Optional
            },
            {
                'method': 'standard',
                'columns': ['numeric_column_normal'],
                'with_mean': True,  # Optional
                'with_std': True     # Optional
            }
        ]
    }
}
```

### Applying Transformations

```python
# Apply transformations
transformed_data = service.transform(data, config)

# The service saves transformers for later use
# You can access them if needed
one_hot_encoder = service.get_transformer('one_hot', ['category_column'])
```

## API Endpoints

The module provides the following API endpoints:

- `POST /api/v1/transformation/transform`: Apply transformations to a dataset
- `POST /api/v1/transformation/upload`: Upload a file and apply transformations

## Testing

Run the tests to verify the functionality:

```bash
pytest backend/tests/test_data_transformation.py -v
```

## Extending the Module

To add a new transformation method:

1. Create a new class in the appropriate submodule (`categorical_encoding` or `feature_scaling`)
2. Implement the required methods: `fit`, `transform`, `fit_transform`, and `get_params`
3. Update the `transformation_service.py` to support the new method
4. Add tests for the new functionality

## Dependencies

- pandas
- numpy
- scikit-learn
- fastapi (for API endpoints)
- python-multipart (for file uploads)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
