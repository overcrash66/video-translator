from voicefixer import VoiceFixer
import inspect

print(f"Init Signature: {inspect.signature(VoiceFixer.__init__)}")
print(f"Restore Signature: {inspect.signature(VoiceFixer.restore)}")
