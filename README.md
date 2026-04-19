# CKKS Encryption + Private Information Retrieval (Legitimate Part)

## Overview

This project implements a secure machine learning inference system using **CKKS (Cheon-Kim-Kim-Song) encryption** for **Private Information Retrieval (PIR)**. The system allows clients to query a server for encrypted model predictions without revealing their data or the model parameters.

## Key Components

### 1. CKKS Encryption
- **Homomorphic Encryption**: Enables computations on encrypted data
- **Approximate Arithmetic**: Supports all floating-point operations for ML applications
- **Batching**: Processes multiple plaintext values in a single ciphertext

### 2. Private Information Retrieval (PIR)
- **Database Encryption**: Stores user data in encrypted form
- **Query Processing**: Client retrieves specific data without revealing queries
- **Security**: Prevents information leakage about client data

## Architecture

```
Client → Encrypted Query → Server
    ↓                       ↓
Encrypted Data        Encrypted Model
    ↓                       ↓
 PIR System           CKKS Encryption
    ↓                       ↓
 Decrypted             Decrypted
   Output                Output
```

## Features

- ✅ **Secure Inference**: No plaintext data exposure
- ✅ **Homomorphic Operations**: Linear algebra on encrypted tensors
- ✅ **PIR Integration**: Private database access
- ✅ **PyTorch Compatibility**: Works with standard PyTorch models
- ✅ **Batch Processing**: Efficient handling of multiple queries

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Microsoft SEAL (for CKKS implementation)
- PIR libraries

### Installation

### Usage

## Security Benefits

1. **Data Privacy**: Client data never leaves encrypted form
2. **Model Protection**: Server cannot access plaintext data
3. **Query Privacy**: PIR ensures client queries remain hidden
4. **Computation Security**: All operations performed on encrypted data

## Applications

- **Healthcare AI**: Medical diagnosis without exposing patient data
- **Financial Services**: Fraud detection with customer privacy
- **Cloud ML**: Secure model inference on third-party servers
- **IoT Analytics**: Edge computing with encrypted sensor data

## References

- [CKKS Scheme](https://eprint.iacr.org/2016/421)
- [Private Information Retrieval](https://en.wikipedia.org/wiki/Private_information_retrieval)
- [Microsoft SEAL](https://github.com/microsoft/SEAL)




