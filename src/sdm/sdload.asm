\ ============================================================================
\ SDMLP loader (main RAM &0900), run by !BOOT.  Finds NBANKS sideways RAM
\ banks (as LPCHECK), loads SDB0.. into them, patching each image's bank
\ numbers first (PNB: the next bank; PB0: bank 0), verifies them all after
\ loading, loads SDM (the rest of the stream) to MAIN0-&7FFF, and starts the
\ player at &8000 in bank 0.
\ ============================================================================

INCLUDE "build/sdm/sdload_cfg.asm"     \ NBANKS, PNB, PB0, MAIN0 (sdmbuild.py)

OSWRCH = &FFEE
OSCLI  = &FFF7
ULA_PG = &FE05
ROMSEL = &F4
bankmap = &0420
lflag = &0432
zp = &70                 \ &70-&77 scratch (LPSTUB re-initialises)
tmp = &70 : tmp2 = &71 : cnt = &72 : idx = &73
ptr = &74                \ (2)
sl = &76 : sh = &77      \ checksum

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
    \ report
    LDX #LO(m_ram) : LDY #HI(m_ram) : JSR prs
    LDY #0
.pr1
    CPY cnt : BEQ pr1d
    LDA cand, Y : JSR phex : LDA #' ' : JSR OSWRCH
    INY
    BNE pr1
.pr1d
    LDX #LO(m_good) : LDY #HI(m_good) : JSR prs
    LDY #0
.pr2
    CPY idx : BEQ pr2d
    LDA good, Y : JSR phex : LDA #' ' : JSR OSWRCH
    INY
    BNE pr2
.pr2d
    LDA idx
    CMP #NBANKS
    BCS enough
    LDX #LO(m_few) : LDY #HI(m_few) : JSR prs
    JMP wait
.enough
    \ load LPB0.. into the first NBANKS good banks, checksumming each
    LDX #LO(m_load) : LDY #HI(m_load) : JSR prs
    LDA #0 : STA idx
.ld
    LDA idx
    CLC : ADC #'0'
    STA cmdb + 8
    JSR OSWRCH
    LDX #LO(cmdb) : LDY #HI(cmdb) : JSR OSCLI
    LDX idx
    LDA good, X
    STA bankmap, X
    LDA good + 1, X           \ the next bank (unused in the last)
    STA &2000 + PNB - &8000
    LDA good
    STA &2000 + PB0 - &8000
    LDX idx
    LDA good, X
    TAX
    SEI
    JSR page
    JSR copy
    JSR csum
    LDX idx
    LDA sl : STA sum_l, X
    LDA sh : STA sum_h, X
    LDA ROMSEL : STA ULA_PG
    CLI
    INC idx
    LDA idx
    CMP #NBANKS
    BNE ld
    \ all loaded: check each again
    LDX #LO(m_ver) : LDY #HI(m_ver) : JSR prs
    LDA #0 : STA idx : STA tmp2
.vl
    LDX idx
    LDA bankmap, X
    TAX
    SEI
    JSR page
    JSR csum
    LDA ROMSEL : STA ULA_PG
    CLI
    LDX idx
    LDA sl : CMP sum_l, X : BNE vbad
    LDA sh : CMP sum_h, X : BNE vbad
    LDA #'.' : JSR OSWRCH
    JMP vn
.vbad
    INC tmp2
    LDA #'X' : JSR OSWRCH
.vn
    INC idx
    LDA idx
    CMP #NBANKS
    BNE vl
    LDA tmp2
    BNE vfail
    LDX #LO(m_ok) : LDY #HI(m_ok) : JSR prs
    JMP go
.vfail
    LDX #LO(m_bad) : LDY #HI(m_bad) : JSR prs
    JSR wait                      \ play anyway: it shows what we hear
.go
    LDX #LO(m_main) : LDY #HI(m_main) : JSR prs
    LDX #LO(m_curs) : LDY #HI(m_curs) : JSR prn  \ cursor off: its flash XORs screen memory, where SDM lands
    LDX #LO(cmdm) : LDY #HI(cmdm) : JSR OSCLI
    SEI
    LDX good
    JSR page
    JMP &8000
.wait
    \ about 6 s to read the screen
    LDX #30
.w1
    LDY #0
.w2
    LDA #0
.w3
    SEC : SBC #1 : BNE w3
    DEY : BNE w2
    DEX : BNE w1
    RTS

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

\ 16-bit sum of &8000-&BFFF into sl/sh
.csum
    LDA #0 : STA sl : STA sh
    STA ptr : LDA #&80 : STA ptr + 1
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
    CMP #&C0
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

.cmdb EQUS "LOAD SDB0 2000", 13
.cmdm EQUS "LOAD SDM", 13
.m_curs EQUB 10, 23, 1, 0, 0, 0, 0, 0, 0, 0, 0
.m_main EQUS "Main RAM part... (BREAK to stop)", 13, 10, 0
.m_title EQUS "LOST PATROL: 1-bit sigma-delta", 13, 10, 0
.m_ram EQUS "RAM banks: ", 0
.m_good EQUS 13, 10, "Unaliased: ", 0
.m_few EQUS 13, 10, "NOT ENOUGH (need ", '0' + NBANKS, ")", 13, 10, 0
.m_load EQUS 13, 10, "Loading ", 0
.m_ver EQUS 13, 10, "Verify ", 0
.m_ok EQUS " all OK", 13, 10, 0
.m_bad EQUS " BAD BANKS", 13, 10, 0
.cand SKIP 16
.good SKIP 16
.sum_l SKIP 16
.sum_h SKIP 16
.end
PRINT "SDLOAD", ~start, "-", ~end
ASSERT end <= &0D00          \ (&0A00 cassette buffer, &0B00 soft keys, &0C00 user characters: unused here; &0D00 is MMFS)
SAVE "build/sdm/sdload.bin", start, end
