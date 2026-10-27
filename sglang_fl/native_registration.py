"""Audit actual dispatcher registration independently of Gems' registrar lists."""

import torch

_EXPECTED = ("ones", "PrivateUse1")
# Library destruction unregisters its kernels. Retain every attempted library,
# including partially initialized ones, for the entire process lifetime.
_libraries = []


class AuditedLibrary:
    def __init__(self, library):
        self._library = library
        self.attempted = []
        self.registered = []

    def __getattr__(self, name):
        return getattr(self._library, name)

    def impl(self, op_name, fn, dispatch_key="", **kwargs):
        entry = (op_name, dispatch_key)
        self.attempted.append(entry)
        if entry != _EXPECTED or len(self.attempted) != 1:
            raise RuntimeError(
                f"Unexpected or duplicate native Gems registration: {entry}"
            )
        result = self._library.impl(op_name, fn, dispatch_key, **kwargs)
        # Gems can swallow the exception from impl. Only a successful return
        # counts, and assert_registered() also rejects swallowed extra attempts.
        self.registered.append(entry)
        return result

    def assert_registered(self):
        if self.attempted != [_EXPECTED] or self.registered != [_EXPECTED]:
            raise RuntimeError(
                "Native Gems ones registration incomplete: "
                f"attempted={self.attempted}, registered={self.registered}"
            )


def enable_ones(flag_gems, **logging_options):
    audited = AuditedLibrary(torch.library.Library("aten", "IMPL"))
    _libraries.append(audited)
    flag_gems.only_enable(lib=audited, include=["ones"], **logging_options)
    audited.assert_registered()
    return audited
