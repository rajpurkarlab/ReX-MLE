"""
Patch to fix RDAgent embedding truncation bug.

The issue: When embedding text exceeds token limits, RDAgent truncates it
and updates kwargs["input_content_list"], but _create_embedding_with_cache
uses the original positional parameter instead of the truncated version.

This causes the retry to still use the original (too-long) text.
"""


def patch_embedding_truncation():
    """Apply patch to fix embedding truncation in RDAgent."""
    import rdagent.oai.backend.base as base_module

    # Store original method
    original_create_embedding_with_cache = base_module.APIBackend._create_embedding_with_cache

    def patched_create_embedding_with_cache(self, input_content_list, *args, **kwargs):
        """
        Patched version that checks kwargs for truncated content.

        When truncation happens in _try_create_chat_completion_or_embedding,
        it sets kwargs["input_content_list"] to the truncated version.
        We need to use that if it exists.
        """
        # Check if kwargs has a (potentially truncated) input_content_list
        # This would be set by the truncation logic at line 497 of base.py
        if "input_content_list" in kwargs:
            input_content_list = kwargs.pop("input_content_list")

        # Call original method with potentially updated input_content_list
        return original_create_embedding_with_cache(self, input_content_list, *args, **kwargs)

    # Apply patch
    base_module.APIBackend._create_embedding_with_cache = patched_create_embedding_with_cache
    print("✓ Applied embedding truncation fix patch")
