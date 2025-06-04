_python_mcompact_memorycli_completion() {
    local IFS=$'
'
    COMPREPLY=( $( env COMP_WORDS="${COMP_WORDS[*]}" \
                   COMP_CWORD=$COMP_CWORD \
                   _PYTHON _M COMPACT_MEMORY.CLI_COMPLETE=complete_bash $1 ) )
    return 0
}

complete -o default -F _python_mcompact_memorycli_completion python -m compact_memory.cli
