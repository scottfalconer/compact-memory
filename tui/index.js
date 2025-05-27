#!/usr/bin/env node
import React, { useState } from 'react';
import { render, Box, Text } from 'ink';
import TextInput from 'ink-text-input';
import SelectInput from 'ink-select-input';
import { exec } from 'child_process';

const Main = () => {
  const [mode, setMode] = useState(null);
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');

  const handleSelect = item => {
    setMode(item.value);
  };

  const handleSubmit = value => {
    if (mode === 'ingest') {
      exec(`python -m gist_memory ingest "${value}"`, (err, stdout, stderr) => {
        setOutput(stdout + stderr);
      });
    } else if (mode === 'query') {
      exec(`python -m gist_memory query "${value}" --top 5`, (err, stdout, stderr) => {
        setOutput(stdout + stderr);
      });
    }
    setInput('');
    setMode(null);
  };

  if (!mode) {
    return (
      <SelectInput
        items={[
          {label: 'Ingest', value: 'ingest'},
          {label: 'Query', value: 'query'},
          {label: 'Exit', value: 'exit'}
        ]}
        onSelect={item => {
          if (item.value === 'exit') process.exit(0);
          handleSelect(item);
        }}
      />
    );
  }

  return (
    <Box flexDirection="column">
      <Box>
        <Text>{mode === 'ingest' ? 'Enter text to ingest:' : 'Enter query:'} </Text>
        <TextInput value={input} onChange={setInput} onSubmit={handleSubmit} />
      </Box>
      {output && (
        <Box marginTop={1} flexDirection="column">
          <Text>{output}</Text>
        </Box>
      )}
    </Box>
  );
};

render(<Main />);
