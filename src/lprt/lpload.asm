\ ============================================================================
\ LPRT loader (main RAM &0900), run by !BOOT.  Finds two sideways RAM banks
\ (as LPCHECK does), loads LPLO (the samples for &0400-&1DFF) and parks it in
\ the second, loads LPH (the player, its tables and the song) into the first
\ and verifies it, loads LPM (the samples for &1E00-&7FFF) where it lives,
\ then, with interrupts off for good, copies LPLO down from the second bank
\ (by a routine at &0300: the copy runs over this loader) and starts the
\ player at &8000.
\ ============================================================================

INCLUDE "build/lprt/lpload_cfg.asm"    \ LOLEN (bytes of LPLO), LO0 (where it goes)

OSWRCH = &FFEE
OSCLI  = &FFF7
ULA_PG = &FE05
ROMSEL = &F4
tmp = &70 : tmp2 = &71 : cnt = &72 : idx = &73
ptr = &74                \ (2)
sl = &76 : sh = &77      \ checksum
TRAMP = &0300

ORG &0900
.start
    LDA #22 : JSR OSWRCH : LDA #6 : JSR OSWRCH
    LDX #LO(m_title) : LDY #HI(m_title) : JSR prs
    SEI
    \ candidates: 15..12, 7..0
    LDA #0 : STA cnt
    LDX #15
.cl
    CPX #12 : BCS cok
    CPX #8 : BCS cnext
.cok
    JSR page
    JSR isram
    BCC cnext
    LDY cnt
    TXA
    STA cand, Y
    INC cnt
.cnext
    DEX
    BPL cl
    \ alias test: tag each candidate, then read the tags back
    LDY #0
.at
    CPY cnt : BEQ atr
    LDX cand, Y
    JSR page
    TYA : ORA #&80 : STA &BFF0
    INY
    BNE at
.atr
    LDY #0 : STY idx
.ar
    CPY cnt : BEQ ard
    LDX cand, Y
    JSR page
    TYA : ORA #&80
    CMP &BFF0
    BNE anext            \ overwritten: an alias of a later one
    TXA
    LDX idx
    STA good, X
    INC idx
.anext
    INY
    BNE ar
.ard
    LDA ROMSEL : STA ULA_PG
    CLI
    LDX #LO(m_ram) : LDY #HI(m_ram) : JSR prs
    LDY #0
.pr2
    CPY idx : BEQ pr2d
    LDA good, Y : JSR phex : LDA #' ' : JSR OSWRCH
    INY
    BNE pr2
.pr2d
    LDA idx
    CMP #2
    BCS enough
    LDX #LO(m_few) : LDY #HI(m_few) : JSR prs
    RTS
.enough
    LDX #LO(m_load) : LDY #HI(m_load) : JSR prs
    \ LPLO -> &2000 -> the second bank
    LDX #LO(cmdl) : LDY #HI(cmdl) : JSR OSCLI
    LDX good + 1
    SEI
    JSR page
    JSR copy
    LDA ROMSEL : STA ULA_PG
    CLI
    LDA #'.' : JSR OSWRCH
    \ LPH -> &2000 -> the first bank, and check it
    LDX #LO(cmdh) : LDY #HI(cmdh) : JSR OSCLI
    JSR csum2
    LDA sl : STA tmp : LDA sh : STA tmp2
    LDX good
    SEI
    JSR page
    JSR copy
    JSR csum
    LDA ROMSEL : STA ULA_PG
    CLI
    LDA sl : CMP tmp : BNE vbad
    LDA sh : CMP tmp2 : BNE vbad
    LDA #'.' : JSR OSWRCH
    JMP go
.vbad
    LDX #LO(m_bad) : LDY #HI(m_bad) : JSR prs
    RTS
.go
    LDX #LO(m_main) : LDY #HI(m_main) : JSR prs
    LDX #LO(m_curs) : LDY #HI(m_curs) : JSR prn  \ cursor off: its flash XORs screen memory, where the samples land
    LDX #LO(cmdm) : LDY #HI(cmdm) : JSR OSCLI
    SEI
    \ the trampoline
    LDY #tramp_end - tramp - 1
.tc
    LDA tramp, Y : STA TRAMP, Y
    DEY
    BPL tc
    LDA good + 1 : STA TRAMP + (t_b1 - tramp) + 1
    LDA good : STA TRAMP + (t_b0 - tramp) + 1
    JMP TRAMP

\ runs at TRAMP: LPLO from the second bank's &8000 to LO0, then the player
.tramp
    LDA #12 : STA ULA_PG
.t_b1
    LDA #0 : STA ULA_PG
    LDX #(LOLEN + 255) DIV 256
    LDY #0
.t_s
    LDA &8000, Y
.t_d
    STA LO0, Y
    INY
    BNE t_s
    INC TRAMP + (t_s - tramp) + 2
    INC TRAMP + (t_d - tramp) + 2
    DEX
    BNE t_s
    LDA #12 : STA ULA_PG
.t_b0
    LDA #0 : STA ULA_PG
    JMP &8000
.tramp_end

\ page bank X (through 12, as the ULA needs from the internal ROMs)
.page
    LDA #12 : STA ULA_PG
    STX ULA_PG
    RTS

\ C set if the paged bank is 16K of RAM and holds no ROM
.isram
    LDY &8007            \ copyright offset: "(C" there means a ROM
    LDA &8000, Y : BNE ir1
    LDA &8001, Y : CMP #'(' : BNE ir1
    LDA &8002, Y : CMP #'C' : BEQ irno
.ir1
    LDA &8000 : STA tmp
    LDA #&55 : STA &8000 : CMP &8000 : BNE irr
    LDA #&AA : STA &8000 : CMP &8000 : BNE irr
    LDA &A000 : PHA
    LDA #&11 : STA &8000
    LDA #&22 : STA &A000
    LDA &8000 : CMP #&11 : BNE irm    \ &A000 shows at &8000: 8K mirrored
    PLA : STA &A000
    LDA tmp : STA &8000
    LDA &BFFF : STA tmp
    LDA #&55 : STA &BFFF : CMP &BFFF : BNE irr2
    LDA #&AA : STA &BFFF : CMP &BFFF : BNE irr2
    LDA tmp : STA &BFFF
    SEC
    RTS
.irm
    PLA : STA &A000
.irr
    LDA tmp : STA &8000
.irno
    CLC
    RTS
.irr2
    LDA tmp : STA &BFFF
    CLC
    RTS

\ copy &2000-&5FFF to &8000-&BFFF
.copy
    LDA #&20 : STA cs + 2
    LDA #&80 : STA cd + 2
    LDX #64
    LDY #0
.cc
.cs LDA &2000, Y
.cd STA &8000, Y
    INY
    BNE cc
    INC cs + 2
    INC cd + 2
    DEX
    BNE cc
    RTS

\ 16-bit sum of &8000-&BFFF (csum) or &2000-&5FFF (csum2) into sl/sh
.csum2
    LDA #&20
    BNE cs0
.csum
    LDA #&80
.cs0
    STA ptr + 1
    CLC : ADC #&40 : STA cnt
    LDA #0 : STA sl : STA sh : STA ptr
    LDY #0
.ck
    LDA (ptr), Y
    CLC : ADC sl : STA sl
    BCC ck1
    INC sh
.ck1
    INY
    BNE ck
    INC ptr + 1
    LDA ptr + 1
    CMP cnt
    BNE ck
    RTS

.prs
    STX ptr : STY ptr + 1
    LDY #0
.ps
    LDA (ptr), Y
    BEQ psd
    JSR OSWRCH
    INY
    BNE ps
.psd
    RTS

\ print X/Y -> a length-prefixed string (may hold zeros)
.prn
    STX ptr : STY ptr + 1
    LDY #0
    LDA (ptr), Y
    STA cnt
.pn1
    INY
    LDA (ptr), Y
    JSR OSWRCH
    DEC cnt
    BNE pn1
    RTS

.phex
    CMP #10
    BCC ph1
    ADC #6
.ph1
    ADC #'0'
    JMP OSWRCH

.cmdl EQUS "LOAD LPLO 2000", 13
.cmdh EQUS "LOAD LPH 2000", 13
.cmdm EQUS "LOAD LPM", 13
.m_curs EQUB 10, 23, 1, 0, 0, 0, 0, 0, 0, 0, 0
.m_main EQUS 13, 10, "Samples... (BREAK to stop)", 13, 10, 0
.m_title EQUS "LOST PATROL: decoded as it plays", 13, 10, 0
.m_ram EQUS "RAM banks: ", 0
.m_few EQUS 13, 10, "NOT ENOUGH (need 2)", 13, 10, 0
.m_load EQUS 13, 10, "Loading ", 0
.m_bad EQUS " BAD BANK", 13, 10, 0
.cand SKIP 16
.good SKIP 16
.end
PRINT "LPLOAD", ~start, "-", ~end
ASSERT end <= &0D00
SAVE "build/lprt/lpload.bin", start, end
