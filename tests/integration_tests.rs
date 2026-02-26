mod test_gguf {
    #[test]
    fn test_ggml_type() {
        use ollama::GgmlType;
        
        assert!(GgmlType::F32.bytes_per_element() == 4);
        assert!(GgmlType::F16.bytes_per_element() == 2);
        assert!(GgmlType::I32.bytes_per_element() == 4);
        assert!(GgmlType::I8.bytes_per_element() == 1);
    }
}

// Add more pure Rust integration tests here as needed
