from unittest.mock import Mock, patch

@patch('src.clients.cohere.Client')
@patch('src.clients.Pinecone')
def test_with_mocked_services(mock_pinecone, mock_cohere):
    # Setup mocks
    mock_cohere.embed.return_value = Mock(embeddings=[[0.1, 0.2, 0.3]])
    
    # Your test code here
    ... 