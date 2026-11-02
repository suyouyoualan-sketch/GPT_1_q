from app.security import decrypt_secret, encrypt_secret


def test_secret_round_trip_and_ciphertext_hides_value() -> None:
    plaintext = "exchange-secret-value"
    encrypted = encrypt_secret(plaintext)
    assert plaintext not in encrypted
    assert decrypt_secret(encrypted) == plaintext

