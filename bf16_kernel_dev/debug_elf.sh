#!/bin/bash
python3 << 'PYEOF'
import struct
with open("/home/bcloud/npu-sandbox/npu-infer/build/qwen3_gemm/mm_128x64x128.o", 'rb') as f:
    ident = f.read(16)
    if ident[:4] != b'\x7fELF':
        print("Not ELF"); exit(1)
    elf_class = ident[4]
    print(f"ELF class: {elf_class} ({'32' if elf_class==1 else '64'})")
    f.seek(0x20)  # e_shoff (ELF32)
    shoff = struct.unpack('<I', f.read(4))[0]
    f.seek(0x2E)  # e_shentsize
    shentsize = struct.unpack('<H', f.read(2))[0]
    f.seek(0x30)  # e_shnum
    shnum = struct.unpack('<H', f.read(2))[0]
    f.seek(0x32)  # e_shstrndx
    shstrndx = struct.unpack('<H', f.read(2))[0]
    print(f"shoff={shoff} shentsize={shentsize} shnum={shnum} shstrndx={shstrndx}")
    
    # Read section header string table
    f.seek(shoff + shstrndx * shentsize + 0x10)
    shstrtab_offset = struct.unpack('<I', f.read(4))[0]
    f.seek(shoff + shstrndx * shentsize + 0x14)
    shstrtab_size = struct.unpack('<I', f.read(4))[0]
    f.seek(shstrtab_offset)
    shstrtab = f.read(shstrtab_size)
    print(f"strtab offset={shstrtab_offset} size={shstrtab_size}")
    print(f"strtab hex[:100]: {shstrtab[:100].hex()}")
    
    # List sections
    for i in range(shnum):
        f.seek(shoff + i * shentsize)
        sh_name_idx = struct.unpack('<I', f.read(4))[0]
        f.seek(shoff + i * shentsize + 4)
        sh_type = struct.unpack('<I', f.read(4))[0]
        # Read sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize
        f.seek(shoff + i * shentsize + 0x10)  # skip to sh_offset
        sh_offset = struct.unpack('<I', f.read(4))[0]
        f.seek(shoff + i * shentsize + 0x14)
        sh_size = struct.unpack('<I', f.read(4))[0]
        
        # Get section name from string table
        if sh_name_idx < shstrtab_size:
            end = shstrtab.find(b'\x00', sh_name_idx)
            if end < 0: end = shstrtab_size
            sh_name = shstrtab[sh_name_idx:end].decode('ascii', errors='replace')
        else:
            sh_name = f"BAD_IDX_{sh_name_idx}"
        
        type_names = {0:'NULL', 1:'PROGBITS', 2:'SYMTAB', 3:'STRTAB', 4:'RELA', 8:'NOBITS', 9:'REL'}
        type_name = type_names.get(sh_type, f'UNK({sh_type})')
        link_info = ""
        if sh_type == 2:  # SYMTAB
            f.seek(shoff + i * shentsize + 0x18)  # sh_link
            sh_link = struct.unpack('<I', f.read(4))[0]
            link_info = f" link={sh_link}"
        print(f"  [{i}] {sh_name:20s} {type_name:8s} offset={sh_offset:6d} size={sh_size:6d}{link_info}")
PYEOF
